import os
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="RAG Chatbot API")

sessions: dict = {}


class AskRequest(BaseModel):
    session_id: str
    question: str
    groq_api_key: str | None = None


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        docs = PyPDFLoader(tmp_path).load()
        os.remove(tmp_path)

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
        ).split_documents(docs)

        retriever = BM25Retriever.from_documents(chunks, k=6)
        session_id = str(uuid.uuid4())
        sessions[session_id] = retriever

        return JSONResponse(content={
            "session_id": session_id,
            "message": f"PDF processed — {len(chunks)} chunks indexed.",
        })
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

        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
        )

        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=sessions[req.session_id],
            llm=llm,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=multi_retriever,
            return_source_documents=True,
        )
        result = qa_chain.invoke({"query": req.question})
        sources = [
            {
                "content": d.page_content,
                "page": d.metadata.get("page", "N/A"),
                "source": d.metadata.get("source", "N/A"),
            }
            for d in result.get("source_documents", [])
        ]
        return JSONResponse(content={"answer": result["result"], "sources": sources})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory="static"), name="static")