from typing import Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from datetime import datetime
import os

app = FastAPI()
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.llms import GPT4All
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore


gpt_4_all_model = "mistral-7b-openorca.gguf2.Q4_0.gguf"
embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
llm = GPT4All(model_name=gpt_4_all_model)


def clean_file_name(file_name: str) -> str:
    return file_name.replace(" ", "_")

@app.post("/api/v1/upload")
async def create_upload_file(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=404, detail="Only PDF files are allowed")
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
    file_name = f"{timestamp}_{file.filename}"
    file_name = clean_file_name(file_name)
    dir_name = file_name.split(".")[0]
    os.makedirs(dir_name, exist_ok=True)
    with open(f"uploads/{dir_name}/{file_name}", "wb") as f:
        f.write(file.file.read())
    return {"file_id": dir_name}


@app.post("/api/v1/chat/{file_id}/ask")
def ask_a_question(file_id: str, q: Union[str, None] = None):
    file_name = clean_file_name(file_id)
    context = None
    file_path=f"uploads/{file_name}/{file_name}.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever()
    return {"file_id": file_id, "question": q, "answer": "42"}  