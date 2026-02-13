import os
import json
import tempfile
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
VECTOR_STORE_PATH = "vector_store"
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")

# Ensure folder exists
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Initialize sentence transformer embeddings (free)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(files, chunk_size=1000, chunk_overlap=100, overwrite=True):
    """Process uploaded files into documents, chunk them, and persist to Chroma vector store.

    This function improves CSV handling by falling back to row-wise parsing when needed,
    ensures text encoding for text files, and supports overwriting the existing vector DB.

    Returns a dict with a brief processing summary.
    """
    from langchain.schema import Document
    import shutil

    docs = []
    file_counts = {}

    for file in files:
        file_ext = os.path.splitext(file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if file_ext == ".pdf":
                # Try LangChain's PyPDFLoader first
                loader = PyPDFLoader(tmp_path)
                loaded = loader.load()

                # If loader returns no usable text, fallback to pypdf (PdfReader)
                if not loaded or all(not (getattr(d, 'page_content', '').strip()) for d in loaded):
                    try:
                        from pypdf import PdfReader
                        pages = []
                        reader = PdfReader(tmp_path)
                        for i, page in enumerate(reader.pages):
                            text = page.extract_text() or ""
                            if text.strip():
                                pages.append(Document(page_content=text, metadata={"source": file.name, "page": i}))
                        if pages:
                            loaded = pages
                    except Exception as e:
                        # If pypdf fails, keep loaded as-is (possibly empty) but log
                        print(f"pypdf fallback failed for {file.name}: {e}")

            elif file_ext == ".csv":
                # Prefer CSVLoader but fall back to pandas row-wise parsing when CSVLoader doesn't return usable text
                try:
                    loader = CSVLoader(tmp_path, encoding="utf-8")
                    loaded = loader.load()
                except Exception:
                    loaded = []

                if not loaded or all(not (getattr(d, 'page_content', '').strip()) for d in loaded):
                    import pandas as pd
                    df = pd.read_csv(tmp_path, dtype=str).fillna("")
                    loaded = []
                    for i, row in df.iterrows():
                        text = " \n".join([f"{col}: {val}" for col, val in row.items()])
                        loaded.append(Document(page_content=text, metadata={"source": file.name, "row": int(i)}))

            elif file_ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                loaded = loader.load()

            else:
                # Unsupported file type – skip
                loaded = []

        except Exception as e:
            # Skip files that fail to load but continue processing others
            print(f"Failed to load {file.name}: {e}")
            loaded = []

        docs.extend(loaded)
        file_counts[file.name] = len(loaded)

    if not docs:
        raise ValueError("No documents were extracted from uploaded files. Check files and loaders.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Optionally overwrite existing vector DB
    if overwrite and os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)

    # Persist to Chroma
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_STORE_PATH)
    try:
        vectordb.persist()
    except Exception:
        # Some older client versions persist on creation; ignore persist errors
        pass

    # Prepare a small preview of the first few documents for UI debugging
    previews = []
    for d in chunks[:5]:
        previews.append({
            "source": d.metadata.get("source") or d.metadata.get("source_file") or "unknown",
            "text_snippet": (d.page_content or "")[:300],
            "metadata": d.metadata,
        })

    return {"files_processed": file_counts, "num_docs": len(docs), "num_chunks": len(chunks), "previews": previews}

def ask_question(query, k=3):
    vectordb = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa({"query": query})

    answer = result["result"]
    sources = result.get("source_documents", [])

    # Save history
    log_result(query, answer, sources)

    return answer, [doc.metadata for doc in sources]

def log_result(query, answer, sources):
    entry = {
        "query": query,
        "answer": answer,
        "sources": [doc.metadata for doc in sources]
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []