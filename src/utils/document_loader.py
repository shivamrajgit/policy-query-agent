"""Document loading and processing utilities."""

import os
import time
import logging
import tempfile
import requests
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and vectorstore creation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_vectorstore_from_urls(self, urls: List[str], request_id: str = "unknown") -> FAISS:
        """
        Create a FAISS vectorstore from a list of URLs.
        Downloads documents from URLs, processes them, and returns a vectorstore.
        """
        all_documents = []
        
        for idx, url in enumerate(urls):
            try:
                documents = self._load_document_from_url(url, request_id)
                all_documents.extend(documents)
                print(f"  URL {idx+1}/{len(urls)} loaded ({len(documents)} pages)")
            except Exception as e:
                print(f"  ERROR loading URL {idx+1}: {e}")
                continue
        
        if not all_documents:
            raise ValueError("No documents could be loaded from the provided URLs")
        
        # Split documents into chunks
        print(f"  Splitting {len(all_documents)} pages into chunks...")
        splits = self.text_splitter.split_documents(all_documents)
        print(f"  Created {len(splits)} chunks")
        
        # Create vectorstore
        print("  Creating embeddings...")
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        return vectorstore
    
    def _load_document_from_url(self, url: str, request_id: str = "unknown") -> List[Document]:
        """
        Load a document from a URL. Currently supports PDF files.
        Downloads the file temporarily and processes it.
        """
        try:
            # Download the file
            download_start = time.perf_counter()
            self.logger.debug(f"[DOWNLOAD] [REQ-{request_id}] Downloading document from {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            download_time = time.perf_counter() - download_start
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            try:
                # Load the document using PyPDFLoader
                parse_start = time.perf_counter()
                self.logger.debug(f"[PARSE] [REQ-{request_id}] Parsing PDF document")
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                parse_time = time.perf_counter() - parse_start
                
                # Add URL metadata to each document
                for doc in documents:
                    doc.metadata.update({
                        "source_url": url,
                        "file_type": ".pdf"
                    })
                
                self.logger.debug(f"[PARSE_DONE] [REQ-{request_id}] Document loaded: Download={download_time:.4f}s, Parse={parse_time:.4f}s, Pages={len(documents)}")
                return documents
            
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            self.logger.error(f"[LOAD_ERROR] [REQ-{request_id}] Error loading document from URL {url}: {e}")
            return []
