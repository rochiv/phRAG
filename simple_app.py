"""
Simplified GraphRAG application that works without Neo4j APOC.
"""
import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1
)

# Initialize Neo4j connection
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

def vector_similarity_search(query: str, limit: int = 5) -> list:
    """Perform vector similarity search in Neo4j.
    
    Args:
        query: User query
        limit: Maximum number of results
        
    Returns:
        List of matching documents
    """
    # Generate embedding for the query
    response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    
    # Execute vector similarity search
    cypher_query = """
    MATCH (d:Document)
    WHERE d.embedding IS NOT NULL
    WITH d, vector.similarity.cosine(d.embedding, $embedding) AS score
    ORDER BY score DESC
    LIMIT $limit
    RETURN d.text AS text, d.source AS source, score
    """
    
    with driver.session() as session:
        result = session.run(cypher_query, {"embedding": embedding, "limit": limit})
        return [dict(record) for record in result]

def execute_graph_query(query: str) -> list:
    """Execute a graph query using an LLM-generated Cypher query.
    
    Args:
        query: User query
        
    Returns:
        Query results
    """
    # Generate Cypher query using LLM
    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert in Neo4j Cypher query generation. 
        Your task is to convert a natural language question into a valid Cypher query.
        
        The database contains Document nodes with the following properties:
        - id: Unique identifier
        - text: Document content
        - source: Source file path
        - file_name: Name of the file
        - embedding: Vector embedding (not to be used in queries directly)
        
        Generate only the Cypher query without any explanation or markdown formatting.
        Use appropriate LIMIT clauses to avoid large result sets.
        """),
        ("human", "{query}")
    ])
    
    cypher_chain = cypher_prompt | llm | StrOutputParser()
    cypher_query = cypher_chain.invoke({"query": query})
    
    # Execute the generated Cypher query
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            return [dict(record) for record in result]
    except Exception as e:
        return [{"error": str(e), "query": cypher_query}]

def classify_question_type(query: str) -> str:
    """Determine if a question is better suited for vector search or graph query.
    
    Args:
        query: User question
        
    Returns:
        "vector" or "graph"
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a question classifier. Determine if a question is better answered using:
        
        1. Vector search: For questions about content, facts, and information retrieval.
           Example: "What are the benefits of GraphRAG?"
        
        2. Graph query: For questions about relationships, counts, and structural information.
           Example: "How many documents are in the database?"
        
        Respond with only one word: either "vector" or "graph".
        """),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query}).strip().lower()
    
    if "vector" in result:
        return "vector"
    else:
        return "graph"

def process_query(query: str) -> str:
    """Process a user query.
    
    Args:
        query: User question
        
    Returns:
        Answer to the question
    """
    try:
        # Classify question type
        question_type = classify_question_type(query)
        
        # Get results based on question type
        if question_type == "vector":
            results = vector_similarity_search(query)
            result_type = "Vector search"
        else:
            results = execute_graph_query(query)
            result_type = "Graph query"
        
        # Generate a response using the results
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on retrieved information."),
            ("human", f"""
            Question: {query}
            
            Retrieved information ({result_type}):
            {results}
            
            Please provide a comprehensive answer to the question based on this information.
            If the information is insufficient, please state so.
            """)
        ])
        
        response_chain = response_prompt | llm | StrOutputParser()
        return response_chain.invoke({})
        
    except Exception as e:
        return f"Error processing your query: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Neo4j GraphRAG (Simple Version)") as demo:
    gr.Markdown("# Neo4j GraphRAG Demo (Simple Version)")
    gr.Markdown("""
    This is a simplified version of the GraphRAG application that works without the Neo4j APOC plugin.
    Ask questions about the documents stored in the knowledge graph.
    """)
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about the documents...",
                lines=3
            )
            submit_btn = gr.Button("Submit")
        
        with gr.Column():
            response_output = gr.Textbox(
                label="Response",
                lines=10,
                interactive=False
            )
    
    submit_btn.click(
        fn=process_query,
        inputs=query_input,
        outputs=response_output
    )
    
    gr.Markdown("""
    ## Example Questions
    - "What are the main topics in the documents?"
    - "Find all documents related to GraphRAG"
    - "How many documents are in the database?"
    """)

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()
    
    # Close Neo4j connection when app is closed
    driver.close() 