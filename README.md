

#  **Medical Chatbot with Generative AI & RAG**  

## 📌 **Project Overview**  
This project is an AI-driven **medical chatbot** leveraging **retrieval-augmented generation (RAG)** to enhance response accuracy by combining **OpenAI’s GPT models** with **Pinecone vector databases**. It efficiently retrieves **medical knowledge** from indexed documents and generates intelligent responses using **LangChain**.  

🔹 **Hybrid Conversational AI**: Integrates GPT-based generation with semantic search.  
🔹 **Dynamic Knowledge Retrieval**: Uses Pinecone for **vector search & embedding-based queries**.  
🔹 **Scalable API-Driven Architecture**: Built using **Flask**, supports easy deployment.  
🔹 **Secure API Key Management**: Uses `.env` for handling **OpenAI & Pinecone** credentials.  
🔹 **Optimized Query Processing**: Embedding-based similarity search for rapid information retrieval.  

---

## ⚙️ **Tech Stack & Architecture**  
### **Core Technologies Used**  
- **Natural Language Processing (NLP)**: OpenAI’s GPT-based LLM for response generation.  
- **Vector Search & Embeddings**: Pinecone **(Fast Approximate Nearest Neighbor Search)**.  
- **Data Preprocessing & Storage**: Sentence Transformers, Text Splitting, and PDF Processing.  
- **Web Framework**: Flask for API-based query handling.  
- **Environment Management**: `.env` for secure credential storage.  

### **System Architecture**  

graph LR
    User -->|Query| Flask API
    Flask API -->|Retrieve Docs| Pinecone Vector DB
    Pinecone Vector DB -->|Retrieve Top-K Chunks| LangChain RAG
    LangChain RAG -->|Generate Response| OpenAI GPT
    OpenAI GPT -->|Return Answer| Flask API
    Flask API -->|Display| Web UI
```

---

## 🚀 **Installation & Setup**  
### **1️⃣ Clone the Repository**  
```bash
git clone <https://github.com/vishwastiwarig/medicalbot_genai>
cd <medicalbot_genai>
```

### **2️⃣ Create an Isolated Environment**  
```bash
conda create -n medibot python=3.10 -y
conda activate medicalbot
```

### **3️⃣ Install Required Dependencies**  
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys & Environment Variables**  
Create a `.env` file in the root directory with:  
```ini
PINECONE_API_KEY = "************"
OPENAI_API_KEY = "************"
```

### **5️⃣ Store Embeddings in Pinecone**  
Before running the chatbot, initialize the **Pinecone Vector Store** with medical knowledge:  
```bash
python store_index.py
```

### **6️⃣ Run the Chatbot Application**  
```bash
python app.py
```
Now, open **http://localhost:8080** in your browser to interact with the AI-powered chatbot.  

---

## 🔬 **Core Functionalities & Innovations**  
### **1️⃣ Retrieval-Augmented Generation (RAG)**  
- Uses **Pinecone’s FAISS-like vector retrieval** to fetch relevant medical data.  
- Embeddings are stored & queried for **semantic similarity search**.  

### **2️⃣ Optimized Document Indexing & Embeddings**  
- Preprocesses **medical research papers, PDFs, and datasets**.  
- Uses **Sentence Transformers** for text chunking and high-dimensional embeddings.  

### **3️⃣ Scalable & Modular API Design**  
- Flask-based **microservices architecture** ensures scalability.  
- API endpoints can be **integrated into healthcare applications**.  

### **4️⃣ Secure Credential Management**  
- **Environment variables (.env)** ensure API keys aren’t exposed in source code.  
- **Pinecone & OpenAI authentication** handled securely.  

---

## 📂 **Project Structure**  
```
📦 Project Root
├── 📁 src/                   # Core ML & NLP modules
│   ├── helper.py             # Embeddings, text preprocessing
│   ├── prompt.py             # Custom AI prompt tuning
│   └── __init__.py
├── 📁 templates/              # HTML front-end
│   ├── chat.html              # UI for chatbot
├── store_index.py             # Pinecone vector store setup
├── app.py                     # Flask API & Chatbot logic
├── requirements.txt           # Dependency list
├── setup.py                   # Python package setup
├── .env                       # API keys (ignored in version control)
├── README.md                  # Documentation
└── LICENSE                    # License file
```

---

## 🚀 **Future Enhancements**  
🔹 **Integration with Voice Assistants (Google Assistant / Alexa)**  
🔹 **Multi-Language Support** (Using Hugging Face Transformers)  
🔹 **Medical Image Processing for Diagnosis Assistance**  
🔹 **Fine-Tuned GPT Model with Medical-Specific Datasets**  

---


## 📜 **License**  
This project is licensed under the **MIT License**. See the **LICENSE** file for details.  

---
