"""
Diagnostic script for testing Neo4j connection.
"""
import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
import socket
import time

# Load environment variables
load_dotenv()

# Get Neo4j connection details
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

def check_host_connectivity(uri):
    """Check if the host in the URI is reachable."""
    try:
        # Extract host and port from URI
        # Handles different URI formats like bolt://, neo4j://, etc.
        parts = uri.split("://")[1].split(":")
        host = parts[0]
        port = int(parts[1].split("/")[0]) if len(parts) > 1 else 7687
        
        print(f"Testing connectivity to {host}:{port}...")
        
        # Try to establish a socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Connection to {host}:{port} successful")
            return True
        else:
            print(f"❌ Cannot connect to {host}:{port} - Is Neo4j running?")
            return False
    except Exception as e:
        print(f"❌ Error checking connectivity: {e}")
        return False

def test_neo4j_connection():
    """Test Neo4j connection with provided credentials."""
    print("\n--- Neo4j Connection Test ---")
    print(f"URI: {neo4j_uri}")
    print(f"Username: {neo4j_username}")
    print(f"Password: {'*' * len(neo4j_password)} (length: {len(neo4j_password)})")
    
    # First check if the host is reachable
    if not check_host_connectivity(neo4j_uri):
        print("\nPossible solutions:")
        print("1. Ensure Neo4j database is running")
        print("2. If using a remote database, check network connectivity")
        print("3. Verify the URI is correct in the .env file")
        return False
    
    # Try to connect to Neo4j
    try:
        print("\nTrying to connect to Neo4j...")
        start_time = time.time()
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        # Verify connection
        driver.verify_connectivity()
        end_time = time.time()
        
        print(f"✅ Successfully connected to Neo4j! ({(end_time - start_time):.2f}s)")
        
        # Try a simple query
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            if record:
                print(f"✅ Test query successful: {record['num']}")
        
        driver.close()
        return True
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        
        if "unauthorized" in str(e).lower() or "authentication" in str(e).lower():
            print("\nAuthentication Error Details:")
            print("1. Make sure username and password are correct")
            print("2. Default username is 'neo4j'")
            print("3. If using Neo4j Desktop, check credentials under 'Manage' → 'Connection Details'")
            print("4. Try resetting the password through Neo4j Desktop or command line")
        
        return False

if __name__ == "__main__":
    test_neo4j_connection() 