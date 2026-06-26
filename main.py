from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import InterviewRAG
import uvicorn

app = FastAPI(title="Interview Prep Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = InterviewRAG()

class QuestionRequest(BaseModel):
    question: str
    topic: str = "general"

class UploadRequest(BaseModel):
    content: str
    filename: str

@app.get("/")
def root():
    return {"status": "Interview Prep Bot is running"}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    try:
        result = rag.answer(req.question, req.topic)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_document(req: UploadRequest):
    try:
        count = rag.add_document(req.content, req.filename)
        return {"message": f"Added {count} chunks from '{req.filename}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics")
def get_topics():
    return {
        "topics": [
            "Python", "Data Structures", "Algorithms", "System Design",
            "Machine Learning", "SQL", "OOP", "Behavioral", "GenAI/LLM", "General"
        ]
    }

@app.delete("/reset")
def reset_knowledge_base():
    rag.reset()
    return {"message": "Knowledge base reset successfully"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)