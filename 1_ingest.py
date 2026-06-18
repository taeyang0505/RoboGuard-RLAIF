import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def build_vector_db():
    if "GOOGLE_API_KEY" not in os.environ or os.environ["GOOGLE_API_KEY"].startswith("여기에"):
        print("[ERROR] GOOGLE_API_KEY is not configured. Set a valid key in the .env file.")
        return

    pdf_path = "./data/ur10e_manual.pdf"
    persist_dir = "./chroma_db"

    if not os.path.exists(pdf_path):
        print(f"[ERROR] Source file not found: '{pdf_path}'. Run 0_download.py first.")
        return

    print("[INFO] Step 1/3 — Loading PDF document (PyPDFLoader).")
    docs = PyPDFLoader(pdf_path).load()

    print("[INFO] Step 2/3 — Splitting document into chunks.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"[INFO]           {len(splits)} text chunks generated.")

    print("[INFO] Step 3/3 — Generating embeddings and building Vector DB.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"[INFO] Vector DB successfully built at '{persist_dir}'.")

if __name__ == "__main__":
    build_vector_db()