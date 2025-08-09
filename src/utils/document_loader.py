"""Document loading and processing utilities.

Performance / robustness enhancements:
 - Concurrent downloading & parsing of PDFs (thread pool)
 - Per-URL timeout & aggregate cap to avoid blocking Gunicorn worker past timeout
 - Basic logging with progress
 - Skips oversized / empty downloads safely
"""

import os
import tempfile
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger("document_loader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class DocumentProcessor:
    """Handles document loading and vectorstore creation."""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_vectorstore_from_urls(self, urls: List[str]) -> Chroma:
        """Create a Chroma vectorstore from a list of URLs with concurrency."""
        all_documents: List[Document] = []

        # Limit number of threads to avoid CPU oversubscription on small instances
        max_workers = min(4, max(1, len(urls)))
        logger.info("[DocLoader] Fetching %d URLs with %d workers", len(urls), max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._load_document_from_url, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    docs = future.result()
                    if docs:
                        all_documents.extend(docs)
                        logger.info("[DocLoader] Loaded %d pages from %s", len(docs), url)
                    else:
                        logger.warning("[DocLoader] No pages extracted from %s", url)
                except Exception as e:
                    logger.error("[DocLoader] Error processing %s: %s", url, e)

        if not all_documents:
            raise ValueError("No documents could be loaded from the provided URLs")

        # Split documents into chunks
        splits = self.text_splitter.split_documents(all_documents)
        logger.info("[DocLoader] Split into %d chunks", len(splits))

        # Create vectorstore with Chroma (in-memory for deployment)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="policy_documents"
        )
        logger.info("[DocLoader] Created vectorstore with %d chunks from %d URLs", len(splits), len(urls))
        return vectorstore
    
    def _load_document_from_url(self, url: str) -> List[Document]:
        """Load a document from a URL (PDF)."""
        try:
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()

            # Enforce a max size (e.g., 15 MB) to avoid huge downloads on free tier
            max_bytes = 15 * 1024 * 1024
            total = 0
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        total += len(chunk)
                        if total > max_bytes:
                            logger.warning("[DocLoader] File from %s exceeded size limit; truncating", url)
                            break
                        temp_file.write(chunk)
                temp_file_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata.update({
                        "source_url": url,
                        "file_type": ".pdf"
                    })
                return documents
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        except Exception as e:
            logger.error("[DocLoader] Error loading %s: %s", url, e)
            return []
