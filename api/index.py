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

app = FastAPI(title="RAG Chatbot API")

sessions: dict = {}

# Paths relative to this file (api/index.py → ../static)
_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_HERE, "..", "static")


class HFInferenceEmbeddings(Embeddings):
    """HuggingFace Inference API embeddings — no PyTorch required."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        # Fixed: use /models/ endpoint (the old /pipeline/feature-extraction/ is deprecated)
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.batch_size = batch_size

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        headers = {}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        response = http_requests.post(
            self.api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            all_embeddings.extend(self._call_api(texts[i: i + self.batch_size]))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]


_embeddings = HFInferenceEmbeddings()


class AskRequest(BaseModel):
    session_id: str
    question: str
    groq_api_key: str | None = None


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(_STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api-config")
async def api_config():
    return JSONResponse(content={"groq_key_configured": bool(os.environ.get("GROQ_API_KEY"))})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
        vectorstore = FAISS.from_documents(chunks, _embeddings)
        os.remove(tmp_path)

        session_id = str(uuid.uuid4())
        sessions[session_id] = vectorstore
        return JSONResponse(content={"session_id": session_id, "message": f"PDF processed — {len(chunks)} chunks indexed."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(req: AskRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")
    try:
        api_key = req.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="No Groq API key provided.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.2),
            chain_type="stuff",
            retriever=sessions[req.session_id].as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )
        result = qa_chain.invoke({"query": req.question})
        sources = [
            {"content": d.page_content, "page": d.metadata.get("page", "N/A"), "source": d.metadata.get("source", "N/A")}
            for d in result.get("source_documents", [])
        ]
        return JSONResponse(content={"answer": result["result"], "sources": sources})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
