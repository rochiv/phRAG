"""
Utilities for ingesting documents into Neo4j.
"""
from typing import List, Dict, Any, Optional
import os
import json
from utils.database import Neo4jDatabase
from utils.document_loader import process_document
from utils.embedding import EmbeddingGenerator

def ingest_document_to_neo4j(
    db: Neo4jDatabase,
    file_path: str,
    embedding_generator: EmbeddingGenerator,
    chunk_size: int = 1000,
    overlap: int = 200,
    node_label: str = "Document",
    embedding_property: str = "embedding"
) -> List[str]:
    """Ingest a document into Neo4j database.
    
    Args:
        db: Neo4jDatabase instance
        file_path: Path to the document file
        embedding_generator: EmbeddingGenerator instance
        chunk_size: Size of document chunks
        overlap: Overlap between chunks
        node_label: Label for document nodes
        embedding_property: Property name for the embedding vector
        
    Returns:
        List of created node IDs
    """
    # Process the document into chunks
    doc_chunks = process_document(file_path, chunk_size, overlap)
    
    # Create embeddings for each chunk
    texts = [chunk["text"] for chunk in doc_chunks]
    embeddings = embedding_generator.generate_batch_embeddings(texts)
    
    # Store nodes in Neo4j
    node_ids = []
    
    for i, chunk in enumerate(doc_chunks):
        # Add embedding to the chunk
        chunk_with_embedding = {
            **chunk,
            embedding_property: embeddings[i]
        }
        
        # Create Cypher query to insert the document chunk
        query = f"""
        CREATE (d:{node_label} {{
            id: $id,
            text: $text,
            source: $source,
            file_name: $file_name,
            chunk_index: $chunk_index,
            total_chunks: $total_chunks,
            {embedding_property}: $embedding
        }})
        RETURN d.id as id
        """
        
        # Prepare parameters
        params = {
            "id": chunk["id"],
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "file_name": chunk["metadata"]["file_name"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "total_chunks": chunk["metadata"]["total_chunks"],
            "embedding": embeddings[i]
        }
        
        # Execute query
        result = db.execute_query(query, params)
        if result and result[0].get("id"):
            node_ids.append(result[0]["id"])
    
    return node_ids

def ingest_directory_to_neo4j(
    db: Neo4jDatabase,
    directory_path: str,
    embedding_generator: EmbeddingGenerator,
    file_extensions: List[str] = [".txt", ".md", ".pdf"],
    chunk_size: int = 1000,
    overlap: int = 200,
    node_label: str = "Document",
    embedding_property: str = "embedding",
    create_index: bool = True
) -> Dict[str, List[str]]:
    """Ingest all documents in a directory into Neo4j database.
    
    Args:
        db: Neo4jDatabase instance
        directory_path: Path to directory containing documents
        embedding_generator: EmbeddingGenerator instance
        file_extensions: List of file extensions to process
        chunk_size: Size of document chunks
        overlap: Overlap between chunks
        node_label: Label for document nodes
        embedding_property: Property name for the embedding vector
        create_index: Whether to create a vector index
        
    Returns:
        Dictionary mapping filenames to lists of created node IDs
    """
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    
    # Create vector index if needed
    if create_index:
        embedding_dim = 1536 if embedding_generator.model_name == "openai" else 384
        db.create_vector_index(
            index_name=f"{node_label.lower()}_{embedding_property}_idx",
            node_label=node_label,
            property_name=embedding_property,
            dimension=embedding_dim
        )
    
    # Process each file in the directory
    results = {}
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        # Check if it's a file with an allowed extension
        if os.path.isfile(file_path) and any(file_name.endswith(ext) for ext in file_extensions):
            print(f"Processing {file_name}...")
            try:
                node_ids = ingest_document_to_neo4j(
                    db=db,
                    file_path=file_path,
                    embedding_generator=embedding_generator,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    node_label=node_label,
                    embedding_property=embedding_property
                )
                results[file_name] = node_ids
                print(f"Successfully ingested {file_name} ({len(node_ids)} chunks)")
            except Exception as e:
                print(f"Error ingesting {file_name}: {e}")
    
    return results 