import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 1. Setup the Page
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Chat with any PDF")

# 2. Sidebar for Inputs
with st.sidebar:
    st.header("Setup")
    
    # Try to get key from secrets, otherwise ask user
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("API Key loaded from secrets! 🔒")
    else:
        api_key = st.text_input("Enter your Groq API Key:", type="password")
        
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# 3. Process the PDF (Only runs if a file is uploaded)
if uploaded_file and api_key:
    try:
        @st.cache_resource
        def process_pdf(file):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name

            # Load and Split
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            # Create Embeddings & Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            
            # Clean up temp file
            os.remove(tmp_path)
            return vectorstore

        # Process the file
        with st.spinner("Processing PDF..."):
            db = process_pdf(uploaded_file)

        # 4. Setup the Brain (Groq Llama-3)
        llm = ChatGroq(
            groq_api_key=api_key, 
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )

        # 5. Connect Brain + Memory
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        # 6. Chat Interface
        st.write("---")
        st.write("### Ask a question about the document:")
        question = st.text_input("Type your question here...")

        if question:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": question})
                st.success(result["result"])
                
                with st.expander("See Source Details"):
                    st.write(result["source_documents"])

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif not api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to start.")
elif not uploaded_file:

    st.info("👈 Please upload a PDF document to start chatting.")
