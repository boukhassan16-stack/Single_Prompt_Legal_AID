import os
import re
import json
from pathlib import Path
from typing import List, Dict

import streamlit as st
from groq import Groq

# Optional RAG dependencies
try:
    import chromadb
    from chromadb.utils import embedding_functions
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

APP_TITLE = "PakLegal AI — Pakistani Legal Aid Assistant"
DEFAULT_MODEL = "openai/gpt-oss-120b"
KNOWLEDGE_DIR = Path("knowledge")
model = st.selectbox(
    "Groq model",
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
    ],
    index=0,
)
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
)

SYSTEM_SAFETY = """
You are an AI legal-information assistant for Pakistan.

Provide general legal information, not legal advice.
Do not claim to be a lawyer or guarantee outcomes.
Do not invent laws, sections, procedures, deadlines, fees, offices,
citations, or URLs.
Prefer retrieved knowledge-base sources.
Clearly identify uncertainty and province/city dependencies.
Use placeholders instead of inventing facts in drafts.
For emergencies or immediate danger, advise contacting appropriate
emergency, law-enforcement, or qualified legal-help services.
"""

TOPIC_PROMPTS = {
    "General legal issue": "Identify the legal issue, relevant Pakistani legal concepts, and safe next steps.",
    "Tenancy dispute": "Analyze a landlord/tenant dispute. Focus on lease terms, rent records, notices, possession/eviction issues, province-specific uncertainty, evidence, and appropriate forums.",
    "Theft / FIR": "Explain the general process around a theft complaint/FIR, evidence preservation, police complaint steps, and what to do if the FIR is not registered. Do not invent criminal-law sections unless supported by retrieved sources.",
    "Online fraud": "Analyze an online fraud/cybercrime complaint. Focus on preserving digital evidence, transaction records, account details, reporting channels, and escalation. Do not invent agency procedures or URLs.",
}

def get_client():
    key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key")
    if not key:
        return None
    return Groq(api_key=key)

def groq_chat(client: Groq, messages: List[Dict], model: str = DEFAULT_MODEL, temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def load_knowledge_documents() -> List[Dict]:
    """
    Loads .txt/.md/.pdf files from knowledge/.
    For production, populate this directory with authoritative Pakistani
    legal material and attach source metadata.
    """
    docs = []
    KNOWLEDGE_DIR.mkdir(exist_ok=True)

    for path in KNOWLEDGE_DIR.rglob("*"):
        if not path.is_file():
            continue

        text = ""
        try:
            if path.suffix.lower() in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif path.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            continue

        if text.strip():
            docs.append({
                "id": str(path),
                "source": path.name,
                "text": text,
            })
    return docs

@st.cache_resource(show_spinner=False)
def build_vector_store():
    if not RAG_AVAILABLE:
        return None

    docs = load_knowledge_documents()
    if not docs:
        return None

    client = chromadb.PersistentClient(path=".chroma")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="paklegal_knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing == 0:
        ids, texts, metas = [], [], []
        for i, doc in enumerate(docs):
            # Chunk documents for retrieval.
            words = doc["text"].split()
            chunk_size = 300
            overlap = 40
            start = 0
            chunk_no = 0
            while start < len(words):
                chunk = " ".join(words[start:start + chunk_size])
                ids.append(f"{i}-{chunk_no}")
                texts.append(chunk)
                metas.append({"source": doc["source"], "path": doc["id"]})
                chunk_no += 1
                start += chunk_size - overlap

        if texts:
            collection.add(ids=ids, documents=texts, metadatas=metas)

    return collection

def retrieve_context(query: str, top_k: int = 2) -> str:
    collection = build_vector_store()

    if collection is None:
        return (
            "NO VERIFIED KNOWLEDGE-BASE EXCERPTS WERE RETRIEVED. "
            "Do not present unsupported legal details as verified law."
        )

    result = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    blocks = []
    max_context_chars = 10000
    current_chars = 0

    for i, doc in enumerate(documents):
        source = (
            metadatas[i].get("source", "Unknown source")
            if i < len(metadatas)
            else "Unknown source"
        )

        block = f"[Source: {source}]\n{doc}"

        remaining = max_context_chars - current_chars

        if remaining <= 0:
            break

        block = block[:remaining]

        blocks.append(block)
        current_chars += len(block)

    return "\n\n".join(blocks)

def run_agents(issue_type: str, user_question: str, facts: str, model: str):
    client = get_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is missing.")

    retrieval_query = f"{issue_type}\n{user_question}\n{facts}"
    context = retrieve_context(retrieval_query)

    researcher = groq_chat(
        client,
        [
            {"role": "system", "content": SYSTEM_SAFETY},
            {"role": "user", "content": f"""
You are the Legal Research Agent.
Topic: {issue_type}

User question:
{user_question}

Facts supplied by user:
{facts}

Retrieved knowledge-base material:
{context}

Produce a concise structured research memo.

Keep the response below 600 words.

Include only:
- Issue
- Relevant legal concepts supported by sources
- Important factual questions
- Evidence/documents
- Possible forum/authority when supported
- Uncertainties
- Source names

Do not repeat the user's facts.
Do not invent citations.
"""}
        ],
        model=model,
    )

    action_guide = groq_chat(
        client,
        [
            {"role": "system", "content": SYSTEM_SAFETY},
            {"role": "user", "content": f"""
You are the Legal Action & Documents Agent.

Research memo:
{researcher}

Create a concise practical action plan.

Keep the response below 500 words.

Use bullet points.

Include:
1. First steps
2. Evidence/documents
3. Possible authority/forum
4. Information to take
5. Escalation options
6. Questions for a lawyer/official

Do not repeat the research memo.
Do not invent legal procedures.
"""}
        ],
        model=model,
    )

    draft = groq_chat(
        client,
        [
            {"role": "system", "content": SYSTEM_SAFETY},
            {"role": "user", "content": f"""
You are the Legal Drafting Agent.

Issue:
{issue_type}

User facts:
{facts}

Research memo:
{researcher}

Prepare a simple complaint/application draft.

Keep the draft below 500 words.

Do not invent facts.
Use [PLACEHOLDER] for missing information.
Use simple language.
Include:
- Recipient
- Subject
- Facts
- Requested action
- Evidence list
- Date/signature placeholders
"""}
        ],
        model=model,
    )

    reviewer = groq_chat(
        client,
        [
            {"role": "system", "content": SYSTEM_SAFETY},
            {"role": "user", "content": f"""
You are the Legal Quality Review Agent.

Review the research, action guide and draft.

Keep your review below 400 words.

Return only:
- Unsupported claims
- Missing facts
- Missing evidence
- Jurisdiction issues
- Safety concerns
- Corrections required

Do not reproduce the original documents.
"""}
        ],
        model=model,
    )

    final_answer = groq_chat(
        client,
        [
            {"role": "system", "content": SYSTEM_SAFETY},
            {"role": "user", "content": f"""
You are the Senior Legal Aid Assistant.

Synthesize the agents' work into a concise answer.

Keep the final answer below 800 words.

Do not reproduce the research memo.
Do not reproduce the review.
Do not repeat large portions of the complaint.

Use these headings:

1. What this appears to be
2. General legal information
3. Recommended next steps
4. Documents/evidence
5. Complaint/application draft
6. Important cautions

Clearly identify information that requires local or professional verification.
"""}
        ],
        model=model,
        temperature=0.1,
    )

    return {
        "context": context,
        "researcher": researcher,
        "action_guide": action_guide,
        "draft": draft,
        "reviewer": reviewer,
        "final": final_answer,
    }

# ---------------- UI ----------------

st.title("⚖️ PakLegal AI")
st.caption("AI-powered legal information and complaint-drafting assistant for Pakistan")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
    "Groq model",
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
    ],
    index=0,
)
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="For local testing you can enter it here. Do not commit API keys to Git.",
    )
    if api_key:
        st.session_state["groq_api_key"] = api_key

  

    st.divider()
    st.info(
        "This prototype provides general legal information. "
        "It is not a substitute for advice from a qualified Pakistani lawyer."
    )

if "result" not in st.session_state:
    st.session_state.result = None

issue_type = st.selectbox("What kind of issue do you need help with?", list(TOPIC_PROMPTS.keys()))

user_question = st.text_area(
    "Describe your legal problem",
    height=160,
    max_chars=5000,
    placeholder=(
        "Example: My landlord is refusing to return my security deposit "
        "after I moved out. I have the tenancy agreement and payment receipts."
    ),
)

facts = st.text_area(
    "Important facts (optional)",
    height=140,
    max_chars=4000,
    placeholder=(
        "Province/city, dates, notices received, amount involved, "
        "documents available, etc."
    ),
)

if st.button("🔎 Analyze my issue", type="primary", use_container_width=True):
    if not (os.getenv("GROQ_API_KEY") or api_key):
        st.error("Please provide your GROQ_API_KEY in the sidebar.")
    elif not user_question.strip():
        st.warning("Please describe your legal problem.")
    else:
        with st.spinner("Running legal research and review agents..."):
            try:
                st.session_state.result = run_agents(
                    issue_type,
                    user_question,
                    facts,
                    model,
                )
            except Exception as exc:
                st.error(f"Could not complete the analysis: {exc}")

result = st.session_state.result

if result:
    st.divider()
    st.subheader("Final Legal Aid Response")
    st.markdown(result["final"])

    with st.expander("📚 Retrieved legal research context"):
        st.text(result["context"])

    with st.expander("🧑‍⚖️ Research Agent"):
        st.markdown(result["researcher"])

    with st.expander("🧭 Action & Documents Agent"):
        st.markdown(result["action_guide"])

    with st.expander("📝 Drafting Agent"):
        st.markdown(result["draft"])

    with st.expander("✅ Quality Review Agent"):
        st.markdown(result["reviewer"])
