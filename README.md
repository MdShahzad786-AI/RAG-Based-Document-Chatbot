# 📄 RAG-Based Document Chatbot

> An AI-powered document intelligence application that allows users to upload multiple PDF documents and ask natural language questions based on their content.

Built using **Python, Streamlit, LangChain, Google Gemini, and Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Live Overview

The **RAG-Based Document Chatbot** is designed to make interacting with PDF documents faster and easier.

Instead of manually searching through long documents, users can:

* Upload one or multiple PDF files
* Process and convert documents into searchable chunks
* Generate vector embeddings using Google Gemini
* Ask questions in natural language
* Retrieve relevant document content
* Generate AI-powered answers based on retrieved context
* View the source document and relevant page numbers
* Reset the session and upload new documents

---

## ✨ Features

### 📂 Multi-PDF Upload

Upload and process multiple PDF documents in a single session.

### 🧠 Retrieval-Augmented Generation

The application follows a RAG pipeline to retrieve relevant information from uploaded documents before generating an answer.

### 🔍 Semantic Search

User questions are matched against document embeddings to retrieve the most relevant content.

### 🤖 Gemini-Powered Answers

Google Gemini is used to generate natural language answers based on the retrieved document context.

### 📚 Source References

The chatbot displays the source PDF and relevant page numbers used to generate the response.

### 🗂️ Document Chunking

Large PDF documents are split into smaller overlapping chunks for efficient retrieval.

### 💬 Conversational Interface

A clean Streamlit chat interface allows users to interact naturally with their uploaded documents.

### 🔄 Session Management

The application maintains uploaded documents, vector database, and chat history during the session.

### 🗑️ Reset Engine

Users can clear the current documents, vector database, and conversation history to start a new session.

---

## 🏗️ How It Works

```text
                 ┌──────────────────┐
                 │   Upload PDFs    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   PyPDFLoader    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Document Chunking│
                 │  Chunk: 1000     │
                 │ Overlap: 200     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Gemini Embeddings│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ In-Memory Vector │
                 │     Database     │
                 └────────┬─────────┘
                          │
                          ▼
                   User Question
                          │
                          ▼
                 ┌──────────────────┐
                 │ Similarity Search│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Relevant Context │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Google Gemini LLM│
                 └────────┬─────────┘
                          │
                          ▼
                    AI Response
                    + Sources
```

---

## 🛠️ Tech Stack

| Technology                         | Purpose                            |
| ---------------------------------- | ---------------------------------- |
| **Python**                         | Core programming language          |
| **Streamlit**                      | Web application and user interface |
| **LangChain**                      | RAG pipeline and LLM integration   |
| **Google Gemini**                  | LLM for answer generation          |
| **Gemini Embeddings**              | Document vector embeddings         |
| **PyPDFLoader**                    | PDF document loading               |
| **RecursiveCharacterTextSplitter** | Document chunking                  |
| **InMemoryVectorStore**            | Temporary vector storage           |
| **python-dotenv**                  | Environment variable management    |

---

## 📁 Project Structure

```text
RAG-Based-Document-Chatbot/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

**`app.py`**
Contains the complete Streamlit application, including:

* PDF upload functionality
* Document processing
* Text chunking
* Gemini embeddings
* Vector database creation
* Similarity search
* RAG-based question answering
* Source and page reference extraction
* Chat interface
* Session management

**`requirements.txt`**
Contains all required Python dependencies.

**`.gitignore`**
Prevents sensitive files and unnecessary local files from being uploaded to GitHub.

**`README.md`**
Project documentation.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MdShahzad786-AI/RAG-Based-Document-Chatbot.git
```

### 2. Navigate to the Project Directory

```bash
cd RAG-Based-Document-Chatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

This project requires a Google Gemini API key.

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Alternatively:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> ⚠️ Never upload your `.env` file or API keys to GitHub.

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in your browser.

---

## 💡 How to Use

1. Launch the application.
2. Upload one or more PDF documents.
3. Click **Process & Start Chatting**.
4. The application will:

   * Load the PDFs
   * Split the content into chunks
   * Generate embeddings
   * Store the embeddings in the vector database
5. Ask questions about the uploaded documents.
6. The system retrieves relevant document chunks.
7. Gemini generates an answer based on the retrieved context.
8. Review the source document and page references.

---

## 🧠 RAG Pipeline

The application uses the following workflow:

### Step 1: Document Loading

PDF files are loaded using:

```python
PyPDFLoader
```

### Step 2: Text Chunking

Documents are split using:

```python
RecursiveCharacterTextSplitter
```

Configuration:

```text
Chunk Size: 1000
Chunk Overlap: 200
```

### Step 3: Embedding Generation

Document chunks are converted into vector embeddings using:

```text
models/gemini-embedding-001
```

### Step 4: Vector Storage

The embeddings are stored using:

```python
InMemoryVectorStore
```

### Step 5: Similarity Search

When a user asks a question, the system retrieves the most relevant document chunks.

### Step 6: Answer Generation

The retrieved context and user question are sent to the Gemini LLM to generate a contextual answer.

### Step 7: Source References

The application extracts the original PDF filename and page numbers from the retrieved documents and displays them with the AI response.

---

## 📸 Application Screenshots

### Upload and Process Documents

Add a screenshot of your upload page here:

```markdown
![Upload Screen](assets/upload-screen.png)
```

### Document Chat Interface

Add a screenshot of your chatbot interface here:

```markdown
![Chat Interface](assets/chat-screen.png)
```

> Create an `assets` folder in your repository and upload your application screenshots there.

---

## 🔐 Security Note

API keys are loaded through environment variables.

Make sure your `.gitignore` includes:

```gitignore
.env
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
```

Never commit:

* API keys
* `.env` files
* Streamlit secrets
* Private credentials

---

## ⚠️ Current Limitations

The current version uses an **in-memory vector store**, which means the document embeddings are temporary and are cleared when the application session ends.

Other possible limitations include:

* Supports PDF documents only
* Retrieval currently uses a limited number of similar chunks
* No persistent vector database
* No user authentication
* No document storage after session termination

---

## 🔮 Future Improvements

Planned improvements include:

* [ ] Persistent vector database using Chroma or FAISS
* [ ] Support for DOCX and TXT files
* [ ] Conversation memory
* [ ] Streaming AI responses
* [ ] Advanced retrieval strategies
* [ ] Hybrid search
* [ ] Metadata filtering
* [ ] Document summarization
* [ ] Authentication and user accounts
* [ ] Chat history persistence
* [ ] Cloud deployment
* [ ] Docker containerization
* [ ] Improved prompt engineering
* [ ] Better citation and source visualization

---

## 🎯 Key Learning Outcomes

Through this project, I gained hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* Large Language Model integration
* LangChain
* Google Gemini API
* Document processing
* Text chunking strategies
* Vector embeddings
* Vector databases
* Semantic similarity search
* Prompt construction
* Streamlit application development
* Environment variable management
* Source attribution in AI applications

---

## 👨‍💻 Author

**Mohammed Shahzad**

Aspiring **AI/ML Engineer** focused on building practical AI applications using Machine Learning, Generative AI, LLMs, and Retrieval-Augmented Generation systems.

GitHub: https://github.com/MdShahzad786-AI

---

## ⭐ Support

If you found this project useful, consider giving the repository a **star ⭐**.

It helps others discover the project and motivates me to continue building and sharing more AI projects.

---

## 📄 License

This project is currently available for educational and portfolio purposes.

## ⚙️ How to Deploy & Run

### Method 1: Streamlit Community Cloud (Free Hosting - Recommended for Portfolios)

Streamlit Community Cloud is the best way to showcase this chatbot to recruiters.

1. **Push the repository to GitHub** (Make sure `.env` is omitted; our `.gitignore` handles this automatically).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **Create app** and select your repository, branch, and main file path (`app.py`).
4. Click **Advanced settings...** -> **Secrets** and paste your API key:
   ```toml
   GEMINI_API_KEY = "your-actual-api-key-here"
   ```
5. Click **Deploy**. Your app will be live on a custom sub-domain in less than a minute!

---

### Method 2: Local Development Setup

If you want to run this application on your local machine, follow these steps:

#### 1. Clone the repository and navigate to it:
```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag-chatbot.git
cd enterprise-rag-chatbot
```

#### 2. Create and activate a Python virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install the optimized requirements list:
```bash
pip install -r requirements.txt
```

#### 4. Configure your environment variables:
Create a `.env` file in the root directory:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

#### 5. Run the application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to start chatting with your documents.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE). Feel free to use and adapt it for your own portfolio!
