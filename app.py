"""
Main application for GraphRAG with Gradio interface.
"""
import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from utils.database import Neo4jDatabase
from utils.graph_rag import create_graph_connection, build_graph_rag_workflow

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1
)

# Initialize the Neo4j connection
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

# Create LangChain graph connection
graph = create_graph_connection(
    uri=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password
)

# Create the GraphRAG workflow
graph_rag_workflow = build_graph_rag_workflow(llm=llm, graph=graph)

def process_query(query: str) -> str:
    """Process a user query through the GraphRAG workflow.
    
    Args:
        query: User's question
    
    Returns:
        Generated response
    """
    try:
        # Initialize state with the question
        state = {"question": query}
        
        # Execute the workflow
        result = graph_rag_workflow.invoke(state)
        
        # Return the response
        return result.get("response", "I couldn't find an answer to your question.")
    except Exception as e:
        return f"Error processing your query: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Neo4j GraphRAG") as demo:
    gr.Markdown("# Neo4j GraphRAG Demo")
    gr.Markdown("""
    This application demonstrates Graph-based Retrieval Augmented Generation (GraphRAG) using Neo4j.
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
    - "Find all documents related to [topic]"
    - "What is the relationship between [entity A] and [entity B]?"
    """)

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch() 