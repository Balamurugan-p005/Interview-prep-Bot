import os
import json
import numpy as np
from groq import Groq
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# We use sentence-transformers for embeddings (free, local)
from sentence_transformers import SentenceTransformer
import faiss

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"  # strong quality, generous free tier

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast

EMBED_DIM = 384  # dimension for all-MiniLM-L6-v2
PERSIST_FILE = "knowledge_base.json"

SYSTEM_PROMPT = """You are an expert Interview Preparation Coach with deep knowledge in software engineering, AI/ML, data science, and behavioral interviews.

Your role:
- Give clear, structured answers to interview questions
- Provide example answers when appropriate
- Mention common follow-up questions
- Rate difficulty: Easy / Medium / Hard
- Keep answers concise but complete (3-6 sentences for technical, STAR format for behavioral)

Always format your response as:
**Answer:** <your answer>
**Example:** <code or example if applicable>
**Follow-up questions:** <2-3 likely follow-ups>
**Difficulty:** Easy / Medium / Hard
"""

class InterviewRAG:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.chunks: List[Dict] = []  # stores text + metadata
        self._load_default_knowledge()
        self._load_persisted()

    def _embed(self, texts: List[str]) -> np.ndarray:
        return embedder.encode(texts, convert_to_numpy=True).astype("float32")

    def _load_default_knowledge(self):
        """Seed the RAG with built-in interview Q&A pairs."""
        default_docs = [
            {
                "text": "What is the difference between a list and a tuple in Python?\nLists are mutable (can be changed) while tuples are immutable (cannot be changed after creation). Lists use square brackets [], tuples use parentheses (). Tuples are faster and can be used as dictionary keys. Use tuples for fixed data like coordinates, lists for collections that change.",
                "source": "Python Basics"
            },
            {
                "text": "Explain OOP concepts: Encapsulation bundles data and methods together and restricts access. Inheritance allows a class to inherit properties from a parent class. Polymorphism lets objects of different classes be treated as the same type. Abstraction hides implementation details and shows only essential features.",
                "source": "OOP"
            },
            {
                "text": "What is RAG (Retrieval Augmented Generation)? RAG combines a retrieval system with an LLM. When a user asks a question, relevant documents are retrieved from a knowledge base using vector similarity search, then those documents are injected into the LLM prompt as context. This grounds the LLM's response in real data and reduces hallucinations.",
                "source": "GenAI/LLM"
            },
            {
                "text": "Explain the difference between SQL JOIN types: INNER JOIN returns rows where there's a match in both tables. LEFT JOIN returns all rows from the left table plus matching rows from right. RIGHT JOIN is the opposite. FULL OUTER JOIN returns all rows from both tables. CROSS JOIN returns cartesian product.",
                "source": "SQL"
            },
            {
                "text": "What is the time complexity of binary search? Binary search has O(log n) time complexity because it halves the search space at each step. It requires the array to be sorted. Space complexity is O(1) for iterative, O(log n) for recursive due to call stack.",
                "source": "Algorithms"
            },
            {
                "text": "Tell me about yourself - Behavioral answer: Use the Present-Past-Future formula. Start with your current role/skills, briefly mention past experience that's relevant, then explain why you're excited about this opportunity. Keep it under 2 minutes, focus on what matters to the interviewer, not your entire life story.",
                "source": "Behavioral"
            },
            {
                "text": "What is a REST API? REST (Representational State Transfer) is an architectural style for APIs. It uses HTTP methods: GET (read), POST (create), PUT/PATCH (update), DELETE (remove). REST APIs are stateless, meaning each request contains all info needed. Resources are identified by URLs.",
                "source": "System Design"
            },
            {
                "text": "Explain overfitting and underfitting in Machine Learning. Overfitting: model learns training data too well including noise, performs poorly on new data. Underfitting: model is too simple to capture patterns. Fix overfitting: more data, regularization (L1/L2), dropout, cross-validation. Fix underfitting: more complex model, better features, reduce regularization.",
                "source": "Machine Learning"
            },
            {
                "text": "What are Python decorators? Decorators are functions that modify the behavior of another function without changing its source code. They use the @syntax. Common use cases: logging, authentication, caching, timing. Example: @staticmethod, @classmethod, @property are built-in decorators.",
                "source": "Python"
            },
            {
                "text": "Describe a time you handled a difficult situation at work - STAR method: Situation (set the context), Task (what was your responsibility), Action (what steps did YOU take), Result (quantify the outcome). Always frame as a learning experience, show ownership, and avoid blaming others.",
                "source": "Behavioral"
            },
        ]
        for doc in default_docs:
            self._add_chunk(doc["text"], doc["source"])

    def _add_chunk(self, text: str, source: str):
        embedding = self._embed([text])
        self.index.add(embedding)
        self.chunks.append({"text": text, "source": source})

    def add_document(self, content: str, filename: str) -> int:
        """Split content into chunks and add to index."""
        words = content.split()
        chunk_size = 150
        overlap = 30
        chunks_added = 0

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                self._add_chunk(chunk, filename)
                chunks_added += 1

        self._persist()
        return chunks_added

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        """Retrieve top-k relevant chunks."""
        if self.index.ntotal == 0:
            return []
        q_embed = self._embed([query])
        distances, indices = self.index.search(q_embed, min(k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(dist)
                })
        return results

    def answer(self, question: str, topic: str = "general") -> Dict:
        """Full RAG pipeline: retrieve -> augment -> generate."""
        # 1. Retrieve
        retrieved = self.retrieve(question, k=4)
        context = "\n\n".join([f"[Source: {r['source']}]\n{r['text']}" for r in retrieved])

        # 2. Augment prompt
        user_prompt = f"""Topic: {topic}
Question: {question}

Relevant context from knowledge base:
{context}

Please answer this interview question based on the context above and your knowledge."""

        # 3. Generate with Groq (Llama 3.3 70B)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer_text = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer_text,
            "sources": [r["source"] for r in retrieved],
            "context_used": len(retrieved)
        }

    def reset(self):
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.chunks = []
        if os.path.exists(PERSIST_FILE):
            os.remove(PERSIST_FILE)
        self._load_default_knowledge()

    def _persist(self):
        with open(PERSIST_FILE, "w") as f:
            json.dump(self.chunks, f)

    def _load_persisted(self):
        if not os.path.exists(PERSIST_FILE):
            return
        with open(PERSIST_FILE) as f:
            saved = json.load(f)
        for item in saved:
            # avoid re-adding default knowledge
            if not any(c["text"] == item["text"] for c in self.chunks):
                self._add_chunk(item["text"], item["source"])