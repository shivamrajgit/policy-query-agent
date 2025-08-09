"""Document loading and processing utilities."""

import os
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
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_vectorstore_from_urls(self, urls: List[str]) -> FAISS:
        """
        Create a FAISS vectorstore from a list of URLs.
        Downloads documents from URLs, processes them, and returns a vectorstore.
        """
        all_documents = []
        
        for url in urls:
            try:
                print(f"Processing URL: {url}")
                documents = self._load_document_from_url(url)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                continue
        
        if not all_documents:
            raise ValueError("No documents could be loaded from the provided URLs")
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(all_documents)
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        print(f"Created vectorstore with {len(splits)} document chunks from {len(urls)} URLs")
        
        return vectorstore
    
    def _load_document_from_url(self, url: str) -> List[Document]:
        """
        Load a document from a URL. Currently supports PDF files.
        Downloads the file temporarily and processes it.
        """
        try:
            # Download the file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            try:
                # Load the document using PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                
                # Add URL metadata to each document
                for doc in documents:
                    doc.metadata.update({
                        "source_url": url,
                        "file_type": ".pdf"
                    })
                
                return documents
            
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            print(f"Error loading document from URL {url}: {e}")
            return []
