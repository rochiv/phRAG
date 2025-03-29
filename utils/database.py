"""
Database utilities for connecting to Neo4j and executing queries.
"""
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

class Neo4jDatabase:
    """Class to handle Neo4j database connections and operations."""
    
    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """Initialize the Neo4j database connection.
        
        Args:
            uri: Neo4j URI. Defaults to NEO4J_URI environment variable.
            username: Neo4j username. Defaults to NEO4J_USERNAME environment variable.
            password: Neo4j password. Defaults to NEO4J_PASSWORD environment variable.
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Verify connection
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j database")
        except Exception as e:
            print(f"Failed to connect to Neo4j database: {e}")
            raise
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return the results.
        
        Args:
            query: Cypher query string
            params: Parameters for the query
            
        Returns:
            List of query results as dictionaries
        """
        if not self.driver:
            self.connect()
            
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def create_vector_index(self, index_name: str, node_label: str, property_name: str, 
                           dimension: int = 1536) -> None:
        """Create a vector index in Neo4j.
        
        Args:
            index_name: Name of the index
            node_label: Label of the node to index
            property_name: Property to create the index on
            dimension: Vector dimension size
        """
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{node_label})
        ON n.{property_name}
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimension},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        self.execute_query(query)
        print(f"Vector index '{index_name}' created or already exists")
        
    def vector_search(self, node_label: str, vector_property: str, 
                      embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Perform a vector similarity search in Neo4j.
        
        Args:
            node_label: Label of the node to search
            vector_property: Property containing the vector
            embedding: Query vector
            limit: Maximum number of results to return
            
        Returns:
            List of nodes with their similarity scores
        """
        query = f"""
        MATCH (n:{node_label})
        WHERE n.{vector_property} IS NOT NULL
        WITH n, vector.similarity.cosine(n.{vector_property}, $embedding) AS score
        ORDER BY score DESC
        LIMIT $limit
        RETURN n, score
        """
        return self.execute_query(query, {"embedding": embedding, "limit": limit}) 