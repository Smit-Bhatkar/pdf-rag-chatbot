import os
import tempfile
import uuid
import requests as http_requests
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# FastAPI app instance (Vercel entry-point)
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Chatbot API")

# In-memory session store: { session_id: FAISS vectorstore }
sessions: dict = {}


# ---------------------------------------------------------------------------
# Lightweight embeddings via HuggingFace Inference API (no PyTorch needed)
# ---------------------------------------------------------------------------
class HFInferenceEmbeddings(Embeddings):
    """Call the free HuggingFace Inference API for embeddings.

    Uses the same sentence-transformers/all-MiniLM-L6-v2 model but avoids
    bundling PyTorch + transformers (~3 GB) locally.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.api_url = (
            f"https://api-inference.huggingface.co/pipeline/"
            f"feature-extraction/{model_name}"
        )
        self.batch_size = batch_size

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Send a batch of texts to the HF Inference API."""
        response = http_requests.post(
            self.api_url,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents, batching if needed."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._call_api([text])[0]


# Shared embedding instance (stateless, no model to load)
_embeddings = HFInferenceEmbeddings()


# ---------------------------------------------------------------------------
# Pydantic request model for /ask
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    session_id: str
    question: str
    groq_api_key: str | None = None  # Optional: falls back to GROQ_API_KEY env var


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api-config")
async def api_config():
    """Tell the frontend whether GROQ_API_KEY is pre-configured server-side."""
    return JSONResponse(content={"groq_key_configured": bool(os.environ.get("GROQ_API_KEY"))})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF, build a FAISS vectorstore, return a session ID."""
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

        # Embeddings & vector store (FAISS — lightweight, no onnxruntime)
        vectorstore = FAISS.from_documents(chunks, _embeddings)

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

        # Resolve API key: prefer request body, fall back to environment variable
        api_key = req.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="No Groq API key provided. Either pass groq_api_key in the request or set the GROQ_API_KEY environment variable.",
            )

        llm = ChatGroq(
            groq_api_key=api_key,
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