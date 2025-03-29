"""
Script to ingest documents into Neo4j database.
"""
import os
import argparse
from dotenv import load_dotenv
from utils.database import Neo4jDatabase
from utils.embedding import EmbeddingGenerator
from utils.ingestion import ingest_directory_to_neo4j

def main():
    """Main function to run document ingestion."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ingest documents into Neo4j")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing documents to ingest")
    parser.add_argument("--node_label", type=str, default="Document",
                        help="Label for document nodes in Neo4j")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of document chunks")
    parser.add_argument("--overlap", type=int, default=200,
                        help="Overlap between document chunks")
    parser.add_argument("--embedding_model", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model to use")
    args = parser.parse_args()
    
    # Initialize database connection
    db = Neo4jDatabase()
    try:
        db.connect()
        print(f"Connected to Neo4j database at {db.uri}")
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(args.embedding_model)
        
        # Ingest documents
        print(f"Ingesting documents from {args.data_dir}...")
        results = ingest_directory_to_neo4j(
            db=db,
            directory_path=args.data_dir,
            embedding_generator=embedding_generator,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            node_label=args.node_label
        )
        
        # Print summary
        total_chunks = sum(len(node_ids) for node_ids in results.values())
        print(f"Ingestion complete. Processed {len(results)} documents with {total_chunks} total chunks.")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main() 