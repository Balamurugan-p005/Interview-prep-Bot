# 🤖 Interview Prep Bot — RAG + Claude AI

A full-stack Interview Preparation Bot using:
- **RAG** (Retrieval Augmented Generation) with FAISS vector store
- **Claude claude-sonnet-4-6** as the LLM
- **sentence-transformers** for local embeddings (free, no OpenAI needed)
- **FastAPI** backend + **React** frontend

---

## 📁 Project Structure

```
interview-prep-bot/
├── backend/
│   ├── main.py           # FastAPI routes
│   ├── rag_pipeline.py   # RAG logic (embed → index → retrieve → generate)
│   └── requirements.txt
└── frontend/
    └── App.jsx           # React UI (run in Claude Artifacts or Vite)
```

---

## 🚀 Setup in VS Code

### Step 1 — Get your Anthropic API Key
1. Go to https://console.anthropic.com
2. Create an API key
3. Copy it

### Step 2 — Set up the Backend

Open VS Code terminal and run:

```bash
# Navigate to backend folder
cd interview-prep-bot/backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key as environment variable
# Windows (PowerShell):
$env:ANTHROPIC_API_KEY = "your-api-key-here"
# Mac/Linux:
export ANTHROPIC_API_KEY="your-api-key-here"

# Run the server
python main.py
```

✅ Backend runs at: http://localhost:8000  
✅ API docs at: http://localhost:8000/docs

### Step 3 — Test the API

Using Thunder Client (VS Code extension) or browser:

```
GET  http://localhost:8000/           → health check
GET  http://localhost:8000/topics     → list of topics
POST http://localhost:8000/ask        → ask a question
POST http://localhost:8000/upload     → add custom knowledge
```

Example POST body for /ask:
```json
{
  "question": "What is the difference between list and tuple in Python?",
  "topic": "Python"
}
```

### Step 4 — Run the Frontend

**Option A: Use the App.jsx directly in Claude.ai Artifacts**
- Paste the App.jsx content into a new Artifact
- It will connect to your local backend at localhost:8000

**Option B: Vite React Project**
```bash
# In a new terminal
npm create vite@latest interview-frontend -- --template react
cd interview-frontend

# Replace src/App.jsx with the provided App.jsx
# Replace src/main.jsx with default

npm install
npm run dev
```
Frontend runs at: http://localhost:5173

---

## ⚙️ How RAG Works in This Project

```
User Question
     ↓
[1] Embed question using sentence-transformers (local)
     ↓
[2] Search FAISS index → top 4 relevant chunks
     ↓
[3] Build prompt: question + retrieved context
     ↓
[4] Claude claude-sonnet-4-6 generates structured answer
     ↓
Response with Answer + Sources + Follow-ups
```

---

## 📚 Adding Your Own Knowledge

1. Go to the **Upload Docs** tab in the UI
2. Paste any text: your notes, resume, job description
3. Give it a name and click Upload
4. The bot now uses your content when answering!

Or via API:
```json
POST /upload
{
  "content": "Your notes or study material here...",
  "filename": "My DSA Notes"
}
```

---

## 🧩 VS Code Extensions to Install

- **Python** (Microsoft) — Python language support
- **Thunder Client** — Test your API endpoints
- **Pylance** — Better Python IntelliSense
- **ES7+ React/Redux** — React snippets for frontend

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv active |
| `ANTHROPIC_API_KEY not set` | Set the env variable before running `python main.py` |
| CORS error in browser | Make sure FastAPI is running, CORS is already configured |
| Frontend can't connect | Check backend is running on port 8000 |
| Slow first load | sentence-transformers downloads model on first run (~90MB) |
