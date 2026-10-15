import os
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration & Models Setup
DATA_DIR = "data"
PERSIST_DIR = "chroma_db"

# Local embeddings model via Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Local LLM via Ollama
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Document Loading & Semantic Text Splitting
def build_vectorstore():
    """Loads all PDFs from data directory, splits semantically, and stores in ChromaDB."""
    print("Loading PDF documents from directory...")
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()

    if not docs:
        raise ValueError(f"No PDF files found in '{DATA_DIR}'. Please add PDFs and try again.")

    print(f"Loaded {len(docs)} document pages.")

    print("Splitting text using Semantic Text Splitter...")
    # SemanticChunker calculates embeddings to determine natural breakpoints between sentences/paragraphs
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} semantic chunks.")

    print("Indexing chunks into local Chroma Vector Database...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return vectorstore

# Load or initialize vector database
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("Loading existing Vector Store from disk...")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    vectorstore = build_vectorstore()

# 3. Retriever Setup (Fetches ONLY relevant chunks & PDF sources)
# Retrieves top 3-4 semantic matches, ensuring only relevant source PDFs are retrieved
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4. Prompts Definition
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the user's question ONLY based on the following context retrieved from the PDF documents.
If the context does not contain enough information to answer the question, state explicitly: "ANSWER NOT FOUND IN CONTEXT".

Context:
{context}

Question:
{question}

Answer:
""")

EVALUATE_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation expert. Evaluate if the provided answer correctly addresses the user question based on the retrieved context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Respond with strictly ONE word:
- "CORRECT" if the answer is accurate and directly derived from the context.
- "INCORRECT" if the answer contains hallucinated/unsupported information or fails to answer properly.
- "NOT_FOUND" if the answer states that information was not found.

Evaluation:
""")

# 5. Core RAG Pipeline with Verification & Retry Loop
def generate_answer_with_retry(question: str, max_retries: int = 2):
    # Step A: Retrieve relevant document chunks (Filters down to only required PDFs)
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print("\nAnswer not found.")
        return

    # Extract text content and list unique source PDF files
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = sorted(list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in retrieved_docs])))

    print(f"\n[Retrieval] Found relevant information in: {', '.join(sources)}")

    attempt = 0
    while attempt <= max_retries:
        print(f"[Generation] Attempt {attempt + 1} of {max_retries + 1}...")

        # Step B: Generate Answer
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question}).strip()

        # Handle explicit missing information response
        if "ANSWER NOT FOUND IN CONTEXT" in answer.upper():
            print("\nAnswer not found.")
            return

        # Step C: Evaluate Generated Answer
        eval_chain = EVALUATE_PROMPT | llm | StrOutputParser()
        eval_result = eval_chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        }).strip().upper()

        if "CORRECT" in eval_result:
            print("\n--- FINAL ANSWER ---")
            print(answer)
            print("\n--- SOURCES ---")
            for src in sources:
                print(f"- {src}")
            return
        elif "NOT_FOUND" in eval_result:
            print("\nAnswer not found.")
            return
        else:
            print(f"[Validation Failed] Answer flagged as inaccurate. Retrying...")
            attempt += 1

    # If retries are exhausted without a passing score
    print("\nAnswer not found.")

# 6. Interactive CLI Execution
if __name__ == "__main__":
    print("\n--- Offline Local RAG Assistant Ready ---")
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        if not user_query:
            continue

        generate_answer_with_retry(user_query)