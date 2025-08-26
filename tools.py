# tools.py
import requests
from bs4 import BeautifulSoup
from Bio import Entrez
Entrez.email = "vishwastiwari222@gmail.com"  # Replace with your actual email
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_INDEX_NAME, EMBEDDING_MODEL, llm, embeddings # Import shared components

def pubmed_search_tool(state):
    """Searches PubMed for the latest medical research abstracts."""
    print("---TOOL: PubMed Search---")
    question = state["question"]
    output_text = f"Recent PubMed Articles for '{question}':\n\n"
    try:
        handle = Entrez.esearch(db="pubmed", term=question, retmax=2)
        record = Entrez.read(handle)
        ids = record["IdList"]
        if not ids:
            return {"tool_output": "No PubMed articles found for this query."}
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
        articles = handle.read().split("\n\n")
        for i, article in enumerate(articles, 1):
            output_text += f"Article {i}:\n{article}\n\n"
    except Exception as e:
        output_text += f"Error fetching PubMed articles: {e}"
    return {"tool_output": output_text}

def rag_retriever_tool(state):
    """Searches the internal Pinecone vector database."""
    print("---TOOL: RAG Retriever---")
    question = state["question"]
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    docs = retriever.invoke(question)
    doc_texts = [doc.page_content for doc in docs]
    output_text = "\n\n".join(doc_texts)
    
    return {"tool_output": output_text}

def web_scraper_tool(state):
    """Scrapes a trusted medical website for patient-friendly information."""
    print("---TOOL: Web Scraper---")
    question = state["question"]
    search_url = f"https://www.mayoclinic.org/search/search-results?q={question.replace(' ', '%20')}"
    print(f"Searching for a relevant article at: {search_url}")

    try:
        response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        first_link = soup.select_one(".list-unstyled .list__item a")
        if not first_link or not first_link.get('href'):
            return {"tool_output": "Could not find a relevant article to scrape."}
        
        article_url = "https://www.mayoclinic.org" + first_link.get('href')
        print(f"Scraping article: {article_url}")

        response = requests.get(article_url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find(id="main-content") or soup.find('article')
        
        if main_content:
            return {"tool_output": main_content.get_text(separator='\n', strip=True)}
        else:
            return {"tool_output": "Failed to extract content from the article."}

    except Exception as e:
        return {"tool_output": f"An error occurred during web scraping: {e}"}

# We need to initialize the shared components here as well
# This is a bit of a workaround for simplicity. In a larger app, you'd use a more robust dependency injection system.
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")