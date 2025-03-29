"""
Utilities for loading and processing documents.
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import hashlib
from unstructured.partition.text import partition_text

def load_text_file(file_path: str) -> str:
    """Load text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as a string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
        
def split_text_into_chunks(text: str, chunk_size: int = 1000, 
                         overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Input text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Simple chunking by character count
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a sentence or paragraph break
        if end < len(text):
            # Look for paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            else:
                # Look for sentence break (period followed by space)
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2
        
        chunks.append(text[start:end])
        start = end - overlap
        
    return chunks
    
def process_document(file_path: str, chunk_size: int = 1000, 
                    overlap: int = 200) -> List[Dict[str, Any]]:
    """Process a document into chunks with metadata.
    
    Args:
        file_path: Path to the document
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of document chunks with metadata
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get file info
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Handle different file types
        if file_ext in ['.txt', '.md']:
            text = load_text_file(file_path)
        else:
            # Use unstructured for other file types
            elements = partition_text(filename=file_path)
            text = "\n\n".join([str(element) for element in elements])
        
        # Split into chunks
        chunks = split_text_into_chunks(text, chunk_size, overlap)
        
        # Create document chunks with metadata
        doc_chunks = []
        for i, chunk in enumerate(chunks):
            # Generate a unique ID for the chunk
            chunk_id = str(uuid.uuid4())
            
            # Create document dict
            doc_chunk = {
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": file_path,
                    "file_name": file_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            doc_chunks.append(doc_chunk)
            
        return doc_chunks
        
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        raise 