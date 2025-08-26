# ingest.py
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from config import pc, PINECONE_INDEX_NAME, EMBEDDING_MODEL, MODEL_DIMENSIONS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import ServerlessSpec

def ingest_data():
    """
    Loads, chunks, embeds, and uploads documents to the Pinecone index.
    This is a one-time or occasional process.
    """
    # 1. Check if index exists, create if not
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=MODEL_DIMENSIONS,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
        # Optional: You could clear the index here if you want to re-ingest
        # index = pc.Index(PINECONE_INDEX_NAME)
        # index.delete(delete_all=True)
        # print("Cleared existing vectors from index.")

    # 2. Load documents from the folder
    print("Loading documents from 'medical_documents' folder...")
    loader = PyPDFDirectoryLoader("medical_documents/")
    documents = loader.load()
    if not documents:
        print("No documents found. Aborting ingestion.")
        return

    # 3. Split documents into chunks
    print(f"Successfully loaded {len(documents)} document(s).")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunked_docs)} chunks.")

    # 4. Generate embeddings and upload to Pinecone
    print("Generating embeddings and upserting to Pinecone... This may take a few minutes.")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("✅ Data ingestion complete!")
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Index stats:", index.describe_index_stats())

if __name__ == "__main__":
    ingest_data()