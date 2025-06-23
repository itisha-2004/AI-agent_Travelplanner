from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import json
import os

db = None

def load_data():
    path = "rag/destination_data.json"
    if not os.path.exists(path):
        print("destination_data.json not found.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return [Document(page_content=item["content"], metadata={"title": item["title"]}) for item in data]
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return []

def get_vector_db():
    global db
    if db is None:
        docs = load_data()
        if docs:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embeddings)
    return db

def retrieve_info(city: str, interests: list) -> str:
    query = f"{city}, {', '.join(interests)}"
    results = db.similarity_search(query, k=3)
    return "\n".join([f"{doc.metadata['title']}: {doc.page_content}" for doc in results])
