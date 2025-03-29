"""
Utilities for generating embeddings from text.
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EmbeddingGenerator:
    """Class to generate embeddings from text using different models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding generator with the specified model.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model = None
        
    def _load_model(self) -> None:
        """Load the embedding model if not already loaded."""
        if not self.model:
            if self.model_name.startswith("sentence-transformers/"):
                self.model = SentenceTransformer(self.model_name)
            # Add other model types as needed
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values as floats
        """
        if self.model_name.startswith("sentence-transformers/"):
            self._load_model()
            embedding = self.model.encode(text)
            return embedding.tolist()
        elif self.model_name == "openai":
            # Use OpenAI embedding model
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_name}")
            
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embeddings, each as a list of floats
        """
        if self.model_name.startswith("sentence-transformers/"):
            self._load_model()
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        elif self.model_name == "openai":
            # Use OpenAI embedding model for batch
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            return [item.embedding for item in response.data]
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_name}") 