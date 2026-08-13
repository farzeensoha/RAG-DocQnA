import os
import tempfile
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from typer import prompt

from db import init_db, get_conn, get_user_by_username, verify_password
from auth import current_user, require_login, require_admin

# Load environment variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add GROQ_API_KEY=your_key_here to your .env file. "
        "Get a free key at https://console.groq.com/keys"
    )

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError(
        "SESSION_SECRET is missing. Add SESSION_SECRET=some_long_random_string to your .env file "
        "(used to sign login session cookies)."
    )

MODEL_NAME = "llama-3.3-70b-versatile"

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None

# Initialize CPU-friendly local embeddings (~80MB RAM) — runs locally, no server call
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize the LLM ONCE at startup instead of on every /ask request
llm = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=512,
)


@app.on_event("startup")
def on_startup():
    init_db()


class QuestionRequest(BaseModel):
    question: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ResolveRequest(BaseModel):
    notes: str = ""


PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")


# ---------- Pages ----------

@app.get("/")
def serve_login_page(request: Request):
    user = current_user(request)
    if user:
        return RedirectResponse("/admin" if user["role"] == "admin" else "/app")
    return FileResponse(os.path.join(PUBLIC_DIR, "login.html"))


@app.get("/app")
def serve_app_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/")
    if user["role"] != "user":
        return RedirectResponse("/admin")
    return FileResponse(os.path.join(PUBLIC_DIR, "app.html"))


@app.get("/admin")
def serve_admin_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/")
    if user["role"] != "admin":
        return RedirectResponse("/app")
    return FileResponse(os.path.join(PUBLIC_DIR, "admin.html"))


# ---------- Auth ----------

@app.post("/login")
def login(req: LoginRequest, request: Request):
    row = get_user_by_username(req.username)
    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    request.session["user"] = {"id": row["id"], "username": row["username"], "role": row["role"]}
    return {"role": row["role"]}


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "logged out"}


@app.get("/me")
def me(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


# ---------- Admin: document upload ----------

@app.post("/admin/upload")
async def upload_doc(file: UploadFile = File(...), user=Depends(require_admin)):
    global retriever
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    if suffix.lower() == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()
    os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    conn = get_conn()
    conn.execute(
        "INSERT INTO documents (filename, uploaded_by, uploaded_at) VALUES (?, ?, ?)",
        (file.filename, user["id"], datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    return {"status": "success", "message": f"Successfully indexed {file.filename}"}


# ---------- User: ask ----------

@app.post("/ask")
async def ask_question(req: QuestionRequest, user=Depends(require_login)):
    global retriever
    if not retriever:
        raise HTTPException(status_code=400, detail="No document has been indexed yet.")

    docs = retriever.invoke(req.question)
    context = "\n\n".join([d.page_content for d in docs])
    print("\n\n===== RETRIEVED CONTEXT =====\n", context)
    prompt = f"""Answer accurately using ONLY the context provided below. If you don't know, say "I cannot find that in the context."

Context:
{context}

Question: {req.question}
Answer:"""
    print("\n\n===== FULL PROMPT SENT TO MODEL =====\n", prompt)
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    answer_text = response.content if hasattr(response, "content") else str(response)

    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO qa_logs (timestamp, user_id, model_used, prompt, output, reviewer_status)
           VALUES (?, ?, ?, ?, ?, 'none')""",
        (datetime.now(timezone.utc).isoformat(), user["id"], MODEL_NAME, req.question, answer_text),
    )
    conn.commit()
    log_id = cur.lastrowid
    conn.close()

    return {"answer": answer_text, "log_id": log_id}


# ---------- User: flag an answer for review ----------

@app.post("/logs/{log_id}/flag")
def flag_log(log_id: int, user=Depends(require_login)):
    conn = get_conn()
    row = conn.execute("SELECT * FROM qa_logs WHERE id = ?", (log_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Log entry not found.")
    if row["user_id"] != user["id"]:
        conn.close()
        raise HTTPException(status_code=403, detail="You can only flag your own questions.")
    conn.execute("UPDATE qa_logs SET reviewer_status = 'pending_review' WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()
    return {"status": "flagged"}


# ---------- User: my flagged answers + admin responses ----------

@app.get("/logs/mine")
def list_my_flagged_logs(user=Depends(require_login)):
    conn = get_conn()
    rows = conn.execute(
        """SELECT * FROM qa_logs
           WHERE user_id = ? AND reviewer_status != 'none'
           ORDER BY timestamp DESC""",
        (user["id"],),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Admin: review queue ----------

@app.get("/admin/logs")
def list_review_queue(user=Depends(require_admin)):
    conn = get_conn()
    rows = conn.execute(
        """SELECT qa_logs.*, users.username AS asked_by
           FROM qa_logs JOIN users ON users.id = qa_logs.user_id
           WHERE qa_logs.reviewer_status = 'pending_review'
           ORDER BY qa_logs.timestamp DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/admin/logs/{log_id}/resolve")
def resolve_log(log_id: int, req: ResolveRequest, user=Depends(require_admin)):
    conn = get_conn()
    row = conn.execute("SELECT * FROM qa_logs WHERE id = ?", (log_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Log entry not found.")
    conn.execute(
        """UPDATE qa_logs
           SET reviewer_status = 'reviewed', reviewed_by = ?, reviewed_at = ?, reviewer_notes = ?
           WHERE id = ?""",
        (user["id"], datetime.now(timezone.utc).isoformat(), req.notes, log_id),
    )
    conn.commit()
    conn.close()
    return {"status": "reviewed"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting local server at http://127.0.0.1:8000")
    # reload=False on purpose: reload=True spawns a second process that re-imports
    # this module (and re-loads the embedding model into memory) almost
    # simultaneously with the first process, which is what exhausted the Windows
    # page file. Restart the server manually after code changes instead.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)