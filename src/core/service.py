"""Main service for policy question answering.

Enhancements added:
 - Simple in-memory cache of vector stores keyed by the list of document URLs (order insensitive)
 - Basic instrumentation / timing logs for each processing phase
 - Graceful reuse of existing vector store if same documents are requested again
"""

from typing import List, Dict, Tuple
import time
import hashlib
import logging
from langchain_community.vectorstores import Chroma
from src.utils.document_loader import DocumentProcessor
from .workflow import PolicyQueryWorkflow

# Basic logger setup (falls back to gunicorn/uvicorn logger handlers if present)
logger = logging.getLogger("policy_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PolicyQueryService:
    """Main service that coordinates document processing and question answering."""
    
    def __init__(self, api_keys: List[str] = None, cache_max_entries: int = 3):
        # Pass api_keys to workflow - if None, workflow will load from .env
        self.workflow = PolicyQueryWorkflow(api_keys=api_keys)
        self.document_processor = DocumentProcessor()
        self._vector_store = None
        self._cache: Dict[str, Chroma] = {}
        self._cache_order: List[str] = []  # simple FIFO/LRU hybrid
        self._cache_max_entries = cache_max_entries

    # --------------- Cache Helpers ---------------
    def _urls_cache_key(self, urls: List[str]) -> str:
        # Order insensitive key using sha256 of sorted URLs
        normalized = sorted([u.strip() for u in urls])
        joined = "\n" + "\n".join(normalized)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str):
        return self._cache.get(key)

    def _cache_put(self, key: str, store: Chroma):
        if key in self._cache:
            # move to end (most recent)
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)
            self._cache[key] = store
            return
        # Evict if needed
        if len(self._cache_order) >= self._cache_max_entries:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = store
        self._cache_order.append(key)
    
    def process_documents_and_answer_questions(self, document_urls: List[str], questions: List[str]) -> List[str]:
        """
        Process documents from URLs and answer questions.
        
        Args:
            document_urls: List of URLs to PDF documents
            questions: List of questions to answer
            
        Returns:
            List of answers corresponding to the questions
        """
        if not document_urls:
            raise ValueError("document_urls must not be empty")
        
        if not questions:
            return []
        
        try:
            overall_start = time.time()
            key = self._urls_cache_key(document_urls)
            cache_hit = False

            # Reuse cached vector store if available
            vector_store = self._cache_get(key)
            if vector_store is not None:
                cache_hit = True
                logger.info("[PolicyService] Cache hit for documents set (%s)" % key[:8])
            else:
                logger.info("[PolicyService] Cache miss. Building vector store for %d URLs" % len(document_urls))
                t0 = time.time()
                vector_store = self.document_processor.create_vectorstore_from_urls(document_urls)
                build_dur = time.time() - t0
                logger.info("[PolicyService] Vector store built in %.2fs" % build_dur)
                self._cache_put(key, vector_store)

            self._vector_store = vector_store

            # Set up the workflow with the vector store (cheap)
            t1 = time.time()
            self.workflow.set_vector_store(vector_store)
            wf_dur = time.time() - t1

            # Answer questions
            t2 = time.time()
            answers = self.workflow.answer_questions(questions)
            answer_dur = time.time() - t2
            total_dur = time.time() - overall_start

            logger.info(
                "[PolicyService] Done. cache_hit=%s build=%.2fs wf=%.2fs answer=%.2fs total=%.2fs questions=%d" % (
                    cache_hit, 0.0 if cache_hit else build_dur, wf_dur, answer_dur, total_dur, len(questions)
                )
            )

            return answers

        except Exception as e:
            logger.exception("[PolicyService] Failure: %s", e)
            raise RuntimeError(f"Failed to process documents and answer questions: {e}")
    
    def answer_questions_with_existing_documents(self, questions: List[str]) -> List[str]:
        """
        Answer questions using previously loaded documents.
        
        Args:
            questions: List of questions to answer
            
        Returns:
            List of answers corresponding to the questions
        """
        if self._vector_store is None:
            raise ValueError("No documents loaded. Please call process_documents_and_answer_questions first.")
        
        return self.workflow.answer_questions(questions)
    
    def load_documents(self, document_urls: List[str]) -> None:
        """
        Load documents from URLs for later use.
        
        Args:
            document_urls: List of URLs to PDF documents
        """
        if not document_urls:
            raise ValueError("document_urls must not be empty")
        
        try:
            vector_store = self.document_processor.create_vectorstore_from_urls(document_urls)
            self._vector_store = vector_store
            self.workflow.set_vector_store(vector_store)
        except Exception as e:
            raise RuntimeError(f"Failed to load documents: {e}")
    
    @property
    def has_loaded_documents(self) -> bool:
        """Check if documents have been loaded."""
        return self._vector_store is not None
