"""Main service for policy question answering."""

import time
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from src.utils.document_loader import DocumentProcessor
from .workflow import PolicyQueryWorkflow


class PolicyQueryService:
    """Main service that coordinates document processing and question answering."""
    
    def __init__(self, api_keys: List[str] = None):
        self.logger = logging.getLogger(__name__)
        # Pass api_keys to workflow - if None, workflow will load from .env
        self.workflow = PolicyQueryWorkflow(api_keys=api_keys)
        self.document_processor = DocumentProcessor()
        self._vector_store = None
    
    def process_documents_and_answer_questions(self, document_urls: List[str], questions: List[str], request_id: str = "unknown") -> List[str]:
        """
        Process documents from URLs and answer questions.
        
        Args:
            document_urls: List of URLs to PDF documents
            questions: List of questions to answer
            request_id: Unique identifier for this request
            
        Returns:
            List of answers corresponding to the questions
        """
        if not document_urls:
            raise ValueError("document_urls must not be empty")
        
        if not questions:
            return []
        
        try:
            # Time document processing
            doc_start = time.perf_counter()
            print("Loading documents...")
            
            # Load and process documents
            vector_store = self.document_processor.create_vectorstore_from_urls(document_urls, request_id)
            self._vector_store = vector_store
            
            doc_time = time.perf_counter() - doc_start
            print(f"Documents processed in {doc_time:.2f}s")
            
            # Set up the workflow with the vector store
            self.workflow.set_vector_store(vector_store)
            
            # Time question answering
            questions_start = time.perf_counter()
            print(f"Processing {len(questions)} questions...")
            
            # Answer questions
            answers = self.workflow.answer_questions(questions, request_id)
            
            questions_time = time.perf_counter() - questions_start
            print(f"Questions processed in {questions_time:.2f}s")
            print(f"Total: Documents={doc_time:.2f}s + Questions={questions_time:.2f}s = {doc_time + questions_time:.2f}s")
            
            return answers
            
        except Exception as e:
            print(f"ERROR: Failed to process documents and answer questions: {e}")
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
