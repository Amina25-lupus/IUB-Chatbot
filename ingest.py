import re
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from huggingface_hub import login

# LangChain & Vector Store Imports
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup & Authentication
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

def clean(text):
    """Removes messy whitespace and excessive newlines."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def create_vector_db():
    all_docs = []
    # Using a relative path to your data folder
    data_path = Path("data")

    if not data_path.exists():
        print(f"❌ Error: The folder '{data_path}' does not exist!")
        return

    # Get list of all .docx files
    doc_files = list(data_path.glob("*.docx"))
    print(f"📂 Found {len(doc_files)} documents in the data folder.")

    if not doc_files:
        print("❌ No .docx files found. Please check your data folder.")
        return

    # 2. Load and Clean
    print("🚀 Loading and cleaning text...")
    for path in tqdm(doc_files, desc="Processing Docs"):
        try:
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
            for doc in docs:
                cleaned_content = clean(doc.page_content)
                # Ensure the document isn't empty after cleaning
                if len(cleaned_content) > 10:
                    doc.page_content = cleaned_content
                    all_docs.append(doc)
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")

    # 3. Split into Chunks
    if not all_docs:
        print("❌ No text extracted from documents.")
        return

    print(f"📄 Splitting {len(all_docs)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✂️ Created {len(chunks)} chunks.")

    # 4. Generate Embeddings and Save
    print("🧠 Starting Embedding process (using Multilingual-E5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("💾 Success! 'faiss_index' has been created.")

if __name__ == "__main__":
    create_vector_db()