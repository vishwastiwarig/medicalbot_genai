import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
PINECONE_INDEX_NAME = "medical-rag-index"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-1.5-flash-latest"
MODEL_DIMENSIONS = 768

# --- API Keys & Client Initialization ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✅ APIs Configured.")
except Exception as e:
    print(f"Error during configuration: {e}")
    exit()

# --- LLM and Embeddings ---
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=1)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)