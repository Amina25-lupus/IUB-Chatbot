# LangChain Modern Imports
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
# # Updated imports for newer LangChain versions
# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_huggingface import HuggingFacePipeline



import os

# LangChain Modern Core and Community Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama  # <-- Modern Ollama Import
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration
DB_FAISS_PATH = 'faiss_index'

def get_qa_chain():
    """
    Sets up a fully local RAG pipeline using Ollama for response generation.
    """
    
    # 1. Load Embeddings (Runs locally on your CPU/GPU)
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
    
    # 2. Load FAISS Vector Database
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Vector database not found at {DB_FAISS_PATH}")
        
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. Initialize Ollama LLM (Fully Local)
    # Ensure you ran 'ollama pull llama3:8b' in your terminal first
    llm = ChatOllama(
        model="gemma2:2b",
        temperature=0.1
    )

    # 4. Define Modern Chat Prompt
    system_prompt = (
        "You are the official AI Assistant for Islamia University of Bahawalpur (IUB). "
        "Use the provided context to answer the student's question accurately. "
        "If the answer is not in the context, say you don't know—do not hallucinate. "
        "Keep the tone professional and helpful. "
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 5. Create the "Stuff Documents" Chain 
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 6. Create the final Retrieval Chain
    qa_chain = create_retrieval_chain(
        db.as_retriever(search_kwargs={'k': 3}), 
        question_answer_chain
    )
    
    return qa_chain

def get_response(query, chain):
    """
    Invokes the local chain and extracts the 'answer' key.
    """
    response = chain.invoke({"input": query})
    return response['answer']