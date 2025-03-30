# phRAG: A Neo4j GraphRAG Proof of Concept

A proof of concept for Graph-based Retrieval Augmented Generation (GraphRAG) using Neo4j, LangChain, and LangGraph.

- TODO: Workaround for simple_app.py

## 📝 Overview

phRAG demonstrates how to build a GraphRAG application that combines:

1. **Document Processing** - Chunking and embedding documents for storage in Neo4j
2. **Neo4j Knowledge Graph** - Storing both document content and relationships with vector search capabilities
3. **Intelligent Query Routing** - Directing questions to either semantic search or graph traversal
4. **LangGraph Workflow** - Orchestrating complex multi-step RAG workflows

This project is based on the concepts described in the [Neo4j GraphRAG workflow article](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/).

## 🛠️ Setup

### Prerequisites

- Python 3.10+
- Neo4j database (local or Neo4j AuraDB)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phRAG.git
cd phRAG
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate phrag
```

3. Create a `.env` file with your credentials:
```bash
cp .env.example .env
# Edit the .env file with your Neo4j and OpenAI credentials
```

### Setting up Neo4j with APOC

The application requires the APOC plugin for Neo4j:

#### For Neo4j Desktop:
1. Open Neo4j Desktop
2. Select your project and database
3. Click on the "Plugins" tab
4. Install the "APOC" plugin
5. Restart your database

#### For Neo4j AuraDB:
APOC is already pre-installed on AuraDB instances.

#### For Neo4j Server:
1. Download the APOC plugin JAR file from [Neo4j Labs releases](https://github.com/neo4j/apoc/releases)
2. Place the JAR file in the `plugins` directory of your Neo4j installation
3. Add the following to your `neo4j.conf` file:
   ```
   dbms.security.procedures.unrestricted=apoc.*
   ```
4. Restart your Neo4j server

### Obtaining API Keys and Tokens

#### OpenAI API Key
1. Visit [OpenAI's platform](https://platform.openai.com/signup) and create an account if you don't have one.
2. Go to the [API Keys section](https://platform.openai.com/api-keys) in your account dashboard.
3. Click "Create new secret key" and give it a name.
4. Copy the generated key and add it to your `.env` file as `OPENAI_API_KEY=your_key_here`.

#### Neo4j Database Setup
1. For cloud hosting (recommended for beginners):
   - Go to [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) and create a free account.
   - Click "New Instance" and select "AuraDB Free" tier.
   - Create a new database and note the connection details (URI, username, password).
   - Add these to your `.env` file.

2. For local installation:
   - Download [Neo4j Desktop](https://neo4j.com/download/) and follow the installation instructions.
   - Create a new local project and database.
   - Set a password and note the connection URI (usually `bolt://localhost:7687`).
   - Add these to your `.env` file.
   
   **Finding Neo4j Local Credentials**:
   - The default username is always `neo4j`
   - For password:
     - If you're using Neo4j Desktop: Open your project, click on the database you created, then click "Manage". Your password is shown under "Connection Details".
     - If you're using Neo4j Server: The password was set during installation or first startup.
     - If you forgot your password: 
       1. In Neo4j Desktop, you can reset it by clicking on the three dots next to your database → Settings → Change Password
       2. For Neo4j Server, stop the server and run: `neo4j-admin set-initial-password newpassword`

#### Hugging Face Token (for deployment only)
1. Visit [Hugging Face](https://huggingface.co/join) and create an account if you don't have one.
2. Go to your profile settings and click on "Access Tokens".
3. Click "New token", give it a name, and set appropriate permissions (at least "write" access).
4. Copy the token and add it to your `.env` file as `HF_TOKEN=your_token_here`.

## 🚀 Usage

### 1. Document Ingestion

To ingest documents into the Neo4j database:

```bash
python ingest.py --data_dir ./data
```

Options:
- `--data_dir`: Directory containing documents to ingest (required)
- `--node_label`: Label for document nodes in Neo4j (default: "Document")
- `--chunk_size`: Size of document chunks (default: 1000)
- `--overlap`: Overlap between document chunks (default: 200)
- `--embedding_model`: Embedding model to use (default: "sentence-transformers/all-MiniLM-L6-v2")

### 2. Running the Application

To run the GraphRAG application locally:

```bash
python app.py
```

This will start a Gradio web interface where you can ask questions about your documents.

### 3. Deploying to Hugging Face Spaces

To deploy the application to Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Configure the Space with your Neo4j and OpenAI credentials as secrets
3. Upload the code to the Space
4. The application will use `huggingface_app.py` as the entry point

## 🧩 Project Structure

```
phRAG/
├── app.py                # Main application with Gradio interface
├── huggingface_app.py    # Hugging Face Spaces compatible version
├── ingest.py             # Document ingestion script
├── environment.yml       # Conda environment definition
├── .env.example          # Example environment variables file
├── data/                 # Sample documents for testing
└── utils/
    ├── database.py       # Neo4j database utilities
    ├── document_loader.py # Document loading and processing
    ├── embedding.py      # Embedding generation
    ├── graph_rag.py      # GraphRAG workflow logic
    └── ingestion.py      # Document ingestion utilities
```

## ⚙️ How It Works

1. **Document Processing**:
   - Documents are split into chunks with metadata
   - Embeddings are generated for each chunk
   - Chunks are stored in Neo4j with their embeddings

2. **Query Processing**:
   - Questions are analyzed to determine the best retrieval strategy
   - For relationship-focused questions, graph traversal is used
   - For semantic queries, vector similarity search is used

3. **Response Generation**:
   - Retrieved information is passed to an LLM
   - The LLM generates a response grounded in the retrieved context

## 📚 Resources

- [Neo4j GraphRAG Workflow with LangChain and LangGraph](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/)
- [LangChain Documentation](https://js.langchain.com/docs/)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Neo4j Vector Search Documentation](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)

## 🚧 Future Improvements

- Adding relationship extraction during document ingestion
- Implementing more advanced query decomposition techniques
- Adding support for multi-hop reasoning
- Creating a visual graph explorer for the knowledge graph
- Adding document upload through the UI

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
