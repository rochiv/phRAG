"""
Utilities for managing the Graph RAG workflow.
"""
import os
from typing import Dict, List, Any, Optional, TypedDict, Union
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import GraphCypherQAChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.graphs import Neo4jGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import langgraph.graph as langgraph
from langgraph.graph import StateGraph, END
from enum import Enum
import json

# Type definitions
class GraphState(TypedDict):
    """Represents the state of our Graph RAG workflow."""
    question: str
    documents: Union[List[Dict[str, Any]], Dict[str, Any], None]
    article_ids: Optional[List[str]]
    response: Optional[str]
    subqueries: Optional[List[str]]
    route: Optional[str]
    cypher_query: Optional[str]

class RouteType(str, Enum):
    """Type of routing for a question."""
    VECTOR_SEARCH = "vector_search"
    GRAPH_QUERY = "graph_query"
    
class SubQuery(TypedDict):
    """Represents a subquery for decomposition."""
    sub_query: str

def create_graph_connection(uri: str, username: str, password: str) -> Neo4jGraph:
    """Create a LangChain Neo4jGraph connection.
    
    Args:
        uri: Neo4j URI
        username: Neo4j username
        password: Neo4j password
        
    Returns:
        Connected Neo4jGraph object
    """
    return Neo4jGraph(
        url=uri,
        username=username, 
        password=password
    )

def create_route_classifier(llm: ChatOpenAI) -> Any:
    """Create a route classifier for questions.
    
    Args:
        llm: LLM to use for classification
        
    Returns:
        Classifier chain
    """
    system_prompt = """
    You are an assistant that routes user questions to the right data source.
    You have two options:
    
    1. vector_search: Use this when the question is seeking factual information or specific details
       that would benefit from a semantic search.
    
    2. graph_query: Use this when the question involves relationships, connections between entities,
       or graph traversals.
    
    Examples of Vector Search:
        - Find information about a specific topic
        - What are the key points in document X?
        - Find articles related to a specific concept
    
    Examples of Graph Query:
        - How many connections exist between X and Y?
        - Find all entities connected to X
        - What is the relationship between X and Y?
        - Count all nodes of type X
    
    Choose the most appropriate route based on the question.
    """
    
    # Create classifier prompt
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # Create structured output
    def parse_route(output: str) -> Dict[str, str]:
        output = output.lower().strip()
        if "vector_search" in output:
            return {"route": RouteType.VECTOR_SEARCH}
        else:
            return {"route": RouteType.GRAPH_QUERY}
    
    # Create the chain
    route_chain = route_prompt | llm | StrOutputParser() | parse_route
    return route_chain

def create_query_decomposer(llm: ChatOpenAI) -> Any:
    """Create a query decomposer chain.
    
    Args:
        llm: LLM to use for decomposition
        
    Returns:
        Query decomposer chain
    """
    system_prompt = """
    You are an expert at breaking down complex questions into simpler subqueries.
    Your task is to analyze the user's question and decompose it into 1-3 simpler search queries.
    These subqueries should:
    
    1. Be self-contained and individually answerable
    2. Together cover all aspects of the original question
    3. Be formulated to get the most relevant information for answering the original question
    
    IMPORTANT: 
    - Return only the list of subqueries, without any explanation or commentary
    - Each subquery should be on a separate line
    - Do not number the subqueries
    - Keep the subqueries simple and direct
    - If the original question is already simple, just return it as is
    """
    
    decompose_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original question: {question}\n\nDecomposed subqueries:")
    ])
    
    def parse_subqueries(output: str) -> Dict[str, List[SubQuery]]:
        subqueries = [line.strip() for line in output.strip().split("\n") if line.strip()]
        return {"subqueries": [{"sub_query": query} for query in subqueries]}
    
    # Create the chain
    decompose_chain = decompose_prompt | llm | StrOutputParser() | parse_subqueries
    return decompose_chain

def create_cypher_generator(llm: ChatOpenAI, graph: Neo4jGraph) -> Any:
    """Create a Cypher query generator chain.
    
    Args:
        llm: LLM to use for generating Cypher
        graph: Neo4jGraph connection
        
    Returns:
        Cypher generator chain
    """
    # Get schema for the graph
    schema = graph.schema
    
    system_prompt = f"""
    You are an expert Neo4j Cypher query generator. Your task is to convert natural language questions
    into valid Cypher queries for a Neo4j database with the following schema:
    
    {schema}
    
    Guidelines:
    1. Generate only the Cypher query, without any explanations or markdown
    2. Make sure the query is valid Cypher syntax
    3. Use parameters with $ syntax for values when appropriate
    4. Limit results to a reasonable number (e.g., LIMIT 10) for large result sets
    5. Include appropriate RETURN clauses that would help answer the user's question
    """
    
    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nCypher query:")
    ])
    
    # Create the chain
    cypher_chain = cypher_prompt | llm | StrOutputParser()
    return cypher_chain

def create_graph_qa_chain(llm: ChatOpenAI, graph: Neo4jGraph) -> GraphCypherQAChain:
    """Create a GraphCypherQAChain.
    
    Args:
        llm: LLM to use for the chain
        graph: Neo4jGraph connection
        
    Returns:
        GraphCypherQAChain instance
    """
    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True
    )

def build_graph_rag_workflow(llm: ChatOpenAI, graph: Neo4jGraph) -> StateGraph:
    """Build the full GraphRAG workflow as a LangGraph.
    
    Args:
        llm: LLM to use in the workflow
        graph: Neo4jGraph connection
        
    Returns:
        StateGraph workflow
    """
    # Create the necessary components
    route_classifier = create_route_classifier(llm)
    query_decomposer = create_query_decomposer(llm)
    cypher_generator = create_cypher_generator(llm, graph)
    graph_qa_chain = create_graph_qa_chain(llm, graph)
    
    # Define the workflow
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    
    # 1. Router node
    def route_question(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        result = route_classifier.invoke({"question": question})
        return {"route": result["route"]}
    
    # 2. Query decomposer node
    def decompose_query(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        result = query_decomposer.invoke({"question": question})
        return {"subqueries": result["subqueries"]}
    
    # 3. Cypher generator node
    def generate_cypher(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        cypher = cypher_generator.invoke({"question": question})
        return {"cypher_query": cypher}
    
    # 4. Graph QA node
    def run_graph_qa(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        result = graph_qa_chain.invoke({"query": question})
        return {"documents": result, "response": result.get("result", "")}
    
    # 5. Response generator node
    def generate_response(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        documents = state.get("documents", {})
        
        if state.get("route") == RouteType.VECTOR_SEARCH:
            subqueries = state.get("subqueries", [])
            subquery_text = "\n".join([f"- {q['sub_query']}" for q in subqueries])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that provides accurate answers based on retrieved documents."),
                ("human", f"""
                Original question: {question}
                
                I broke this down into the following subqueries:
                {subquery_text}
                
                Retrieved information:
                {json.dumps(documents, indent=2)}
                
                Please provide a comprehensive answer to the original question based on this information.
                """)
            ])
            
            response = prompt | llm | StrOutputParser()
            result = response.invoke({})
            
        else:  # Graph query
            # Use the result from graph QA
            result = documents.get("result", "No results found")
            
        return {"response": result}
    
    # Add nodes to the graph
    workflow.add_node("router", route_question)
    workflow.add_node("decomposer", decompose_query)
    workflow.add_node("cypher_generator", generate_cypher)
    workflow.add_node("graph_qa", run_graph_qa)
    workflow.add_node("response_generator", generate_response)
    
    # Add edges
    workflow.add_edge("router", "decomposer", condition=lambda state: state.get("route") == RouteType.VECTOR_SEARCH)
    workflow.add_edge("router", "cypher_generator", condition=lambda state: state.get("route") == RouteType.GRAPH_QUERY)
    workflow.add_edge("decomposer", "response_generator")
    workflow.add_edge("cypher_generator", "graph_qa")
    workflow.add_edge("graph_qa", "response_generator")
    workflow.add_edge("response_generator", END)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    return workflow 