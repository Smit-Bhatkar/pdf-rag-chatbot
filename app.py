import os
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# FastAPI app instance (Vercel entry-point)
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Chatbot API")

# In-memory session store: { session_id: Chroma vectorstore }
sessions: dict = {}

# Shared embedding model (loaded once)
_embeddings = None


def get_embeddings():
    """Lazy-load the embedding model so cold starts are faster."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Pydantic request model for /ask
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    session_id: str
    question: str
    groq_api_key: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF, build a Chroma vectorstore, return a session ID."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name  # Fixed bug: was tmp_[file.name]

        # Load & split
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Embeddings & vector store
        embeddings = get_embeddings()
        vectorstore = Chroma.from_documents(chunks, embeddings)

        # Clean up temp file
        os.remove(tmp_path)

        # Store under a new session ID
        session_id = str(uuid.uuid4())
        sessions[session_id] = vectorstore

        return JSONResponse(
            content={
                "session_id": session_id,
                "message": f"PDF processed successfully — {len(chunks)} chunks indexed.",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(req: AskRequest):
    """Answer a question against a previously uploaded PDF."""
    if req.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a PDF first.",
        )

    try:
        vectorstore = sessions[req.session_id]

        llm = ChatGroq(
            groq_api_key=req.groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        result = qa_chain.invoke({"query": req.question})

        sources = []
        for doc in result.get("source_documents", []):
            sources.append(
                {
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "N/A"),
                }
            )

        return JSONResponse(
            content={
                "answer": result["result"],
                "sources": sources,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory="static"), name="static")