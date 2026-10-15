# Offline Local RAG Assistant

An offline, privacy-focused AI assistant that allows you to chat with your local PDF documents using Retrieval-Augmented Generation (RAG). This project leverages Ollama to run everything locally on your machine, ensuring your data never leaves your computer.

## 🚀 Features

- **Fully Offline**: No API keys or internet connection required.
- **Semantic Chunking**: Uses embeddings to split documents at natural meaning boundaries rather than arbitrary character limits.
- **Self-Correction Loop**: Implements a verification step where the LLM evaluates its own answer against the retrieved context to reduce hallucinations.
- **Source Attribution**: Clearly lists which PDF files were used to generate the response.
- **Persistent Storage**: Uses ChromaDB to index your documents once and reuse them across sessions.

## 🛠️ Tech Stack

- **LLM Framework**: [LangChain](https://www.langchain.com/)
- **Local LLM & Embeddings**: [Ollama](https://ollama.com/)
  - Chat Model: `llama3.2`
  - Embedding Model: `mxbai-embed-large`
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **PDF Processing**: `PyPDFLoader`

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

1. **Ollama**: Download and install from [ollama.com](https://ollama.com/).
2. **Required Models**: Pull the models used in the project via your terminal:
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```
3. **Python 3.10+**

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "Offline Ai Assistant With RAG"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your documents**:
   Place all the PDF files you want to chat with in the `data/` folder.

## 🏃 Usage

Run the main script to start the interactive CLI:

```bash
python main.py
```

- **Asking Questions**: Simply type your query. The assistant will retrieve the relevant context from your PDFs and generate a verified answer.
- **Exiting**: Type `exit` or `quit` to close the application.

## ⚙️ How it Works

1. **Ingestion**: The system scans the `data/` folder for PDFs.
2. **Embedding**: Text is converted into numerical vectors using `mxbai-embed-large`.
3. **Storage**: Vectors are stored in `chroma_db/` for efficient similarity searching.
4. **Retrieval**: The user's query is embedded, and the top 4 most similar chunks are retrieved.
5. **Generation & Verification**:
   - The LLM generates an answer based on the chunks.
   - A secondary evaluation prompt checks the answer for accuracy.
   - If incorrect, the system retries the generation.

## 📂 Project Structure

```text
.
├── data/               # Place your PDF documents here
├── chroma_db/          # Local vector store (automatically created)
├── main.py             # Core RAG pipeline and CLI
└── requirements.txt   # Python dependencies
```
