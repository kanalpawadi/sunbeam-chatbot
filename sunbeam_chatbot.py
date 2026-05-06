import streamlit as st
from dotenv import load_dotenv
import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# ── Session State ──────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if st.session_state.current_chat not in st.session_state.chats:
    st.session_state.chats[st.session_state.current_chat] = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ── Login Page ─────────────────────────────────────────────────────────────────
def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Sunbeam Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == "sunbeam123" and password == "sunbeam@1810":
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")


if not st.session_state.logged_in:
    login_page()
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("User Menu")
    if st.button("New Chat"):
        chat_id = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[chat_id] = []
        st.session_state.current_chat = chat_id
        st.rerun()

    st.divider()
    st.markdown("### Chats")
    with st.container(height=170):
        for chat_id in st.session_state.chats:
            if st.button(chat_id, key=chat_id):
                st.session_state.current_chat = chat_id
                st.rerun()

    if st.button("Logout", type="primary"):
        st.session_state.logged_in = False
        st.session_state.chats = {}
        st.session_state.current_chat = None
        st.rerun()


# ── ChromaDB ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_chromadb():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "knowledge_base")
    db = chromadb.PersistentClient(path=DB_PATH)
    return db.get_or_create_collection("Sunbeam_Data")

collection = load_chromadb()


# ── Embedding Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

embed_model = load_embed_model()


# ── Groq Client ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_groq_client():
    return Groq(api_key=api_key)

groq_client = load_groq_client()


# ── Search ChromaDB ────────────────────────────────────────────────────────────
def search_knowledge_base(query: str) -> str:
    query_embedding = embed_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )
    if not results["documents"] or not results["documents"][0]:
        return "No relevant data found."

    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta.get("source", "Sunbeam")
        context += f"\n[{source}]\n{doc}\n"
    return context


# ── Ask Groq ──────────────────────────────────────────────────────────────────
def ask_groq(user_question: str) -> str:
    context = search_knowledge_base(user_question)

    prompt = f"""You are Sunbeam Institute of Information Technology's official AI assistant.

STRICT RULES:
1. Answer ONLY using the SUNBEAM DATA provided below.
2. Do NOT say "No data found" if relevant data exists — use what is provided.
3. Treat "Sunbeam Infotech", "Sunbeam Institute", and "Sunbeam" as the same organization.
4. Give direct, complete, and well-structured answers.
5. If the question is completely unrelated to Sunbeam, reply: "Please ask a question related to Sunbeam Institute."
6. If truly no relevant data exists in the provided context, say: "Sorry, I don't have that information."
7. Never make up information not present in the data.

SUNBEAM DATA:
{context}

USER QUESTION:
{user_question}

ANSWER:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Chat UI ────────────────────────────────────────────────────────────────────
current_chat = st.session_state.current_chat
messages = st.session_state.chats[current_chat]

st.title("Sunbeam Chatbot")
st.markdown("""
    <style>
    div.stButton > button {
        height: 60px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("About Sunbeam", type="primary", use_container_width=True):
        st.session_state.pending_query = "Give me full information about Sunbeam Institute of Information Technology"
with col2:
    if st.button("Core Java Course", type="primary", use_container_width=True):
        st.session_state.pending_query = "Give me full information of Core Java course"
with col3:
    if st.button("Internships", type="primary", use_container_width=True):
        st.session_state.pending_query = "Give me full information about internship programs provided at Sunbeam."
with col4:
    if st.button("Location", type="primary", use_container_width=True):
        st.session_state.pending_query = "Give me full information about the location of Sunbeam Institute."
with col5:
    if st.button("All Courses", type="primary", use_container_width=True):
        st.session_state.pending_query = "Give me a list of all courses provided at Sunbeam Institute."


# ── Display existing messages ──────────────────────────────────────────────────
for chat in messages:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])


# ── Handle Input ───────────────────────────────────────────────────────────────
user_question = st.chat_input("Ask about Sunbeam...")

if st.session_state.pending_query:
    user_question = st.session_state.pending_query
    st.session_state.pending_query = None

if user_question:
    messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_groq(user_question)
        st.markdown(answer)

    messages.append({"role": "assistant", "content": answer})