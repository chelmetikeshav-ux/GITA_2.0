import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ingest import build_vector_db, load_documents, DB_PATH, EMBEDDING_MODEL

# 1. Configuration: works locally (.env) and on Streamlit Cloud (Secrets)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

MODEL_NAME = "openai/gpt-oss-120b"  # fallback if needed: "llama-3.3-70b-versatile"

# 2. Page configuration
st.set_page_config(page_title="Dialogue with Madhav", page_icon="🪶", layout="centered")

# --- UI/UX: PEACOCK FEATHER ENGINE ---
theme_html = """
<style>
    .stAppViewContainer {
        background: linear-gradient(135deg, #0b0518 0%, #120317 50%, #1f0d02 100%) !important;
        overflow-x: hidden;
    }
    .main-header {
        text-align: center;
        font-family: 'Georgia', serif;
        color: #FFD700;
        text-shadow: 0px 4px 15px rgba(255, 215, 0, 0.5);
        font-size: 2.8rem;
        margin-top: 10px;
        margin-bottom: 0px;
    }
    .sub-header {
        text-align: center;
        font-style: italic;
        color: #fce38a;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    div[data-testid="stForm"], .stTextInput > div {
        border-color: #d4af37 !important;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.15) !important;
    }
    .peacock-feather {
        position: fixed;
        user-select: none;
        pointer-events: none;
        z-index: 1;
        bottom: -120px;
        width: 35px;
        height: 60px;
        background: linear-gradient(to top, rgba(212,175,55,0) 0%, rgba(34,139,34,0.4) 60%, rgba(212,175,55,0.7) 100%);
        border-radius: 50% 50% 50% 50% / 40% 40% 60% 60%;
        box-shadow: inset 0 0 10px rgba(0,128,128,0.5);
        animation: floatUp 14s linear infinite;
    }
    .peacock-feather::after {
        content: '';
        position: absolute;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        width: 16px;
        height: 22px;
        background: radial-gradient(circle, #00ffff 0%, #000080 50%, #4b0082 100%);
        border-radius: 50%;
        box-shadow: 0 0 8px #00ffff;
    }
    @keyframes floatUp {
        0%   { transform: translateY(0) rotate(0deg) translateX(0); opacity: 0; }
        10%  { opacity: 0.5; }
        90%  { opacity: 0.5; }
        100% { transform: translateY(-120vh) rotate(360deg) translateX(50px); opacity: 0; }
    }
    .pf1 { left: 8%;  animation-duration: 15s; animation-delay: 0s; }
    .pf2 { left: 28%; animation-duration: 22s; animation-delay: 4s; }
    .pf3 { left: 48%; animation-duration: 18s; animation-delay: 1s; }
    .pf4 { left: 68%; animation-duration: 25s; animation-delay: 7s; }
    .pf5 { left: 88%; animation-duration: 16s; animation-delay: 3s; }
</style>

<div class="peacock-feather pf1"></div>
<div class="peacock-feather pf2"></div>
<div class="peacock-feather pf3"></div>
<div class="peacock-feather pf4"></div>
<div class="peacock-feather pf5"></div>

<h1 class="main-header">🐚Speaking with Sri Krishna🦚🪈</h1>
<p class="sub-header">Stop trying to fight this alone. Hand your doubts over to Keshav and let His perspective bring you peace</p>
"""
st.markdown(theme_html, unsafe_allow_html=True)


# 3. Load the vector database (auto-builds it if missing, empty, or unreadable)
@st.cache_resource(show_spinner="Preparing the Gita library...")
def load_rag_system():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def open_store():
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    # Rebuild if the database is missing, unreadable, or doesn't match the JSON data
    # (e.g. an old database left over from a previous dataset)
    expected_count = len(load_documents())
    needs_rebuild = False
    try:
        vector_store = open_store()
        if vector_store._collection.count() != expected_count:
            needs_rebuild = True
    except Exception:
        needs_rebuild = True

    if needs_rebuild:
        build_vector_db(embedding_model)
        vector_store = open_store()

    return vector_store.as_retriever(search_kwargs={"k": 3})


try:
    retriever = load_rag_system()
except Exception as e:
    st.error(f"⚠️ Could not load or build the verse database: {e}")
    st.info("Make sure the 'gita_chapters' folder with your .json files is in the repository.")
    st.stop()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


SYSTEM_PROMPT = (
    "You are Lord Krishna (addressed affectionately as Keshav, Madhav, or Achyuta). "
    "The user is coming to you with an intense internal dilemma or emotional pain. "
    "You must speak to them directly in the FIRST PERSON ('I', 'Me', 'My').\n\n"

    "CRITICAL FORMATTING INSTRUCTIONS FOR USER WELL-BEING:\n"
    "- Keep your entire response UNDER 150 words total.\n"
    "- Write in short, punchy sentences (under 12 words per sentence).\n"
    "- Use single-sentence paragraphs to create maximum breathing space.\n"
    "- Do not ramble or use heavy academic language. Speak with ultimate simplicity and gentle love.\n"
    "- Base your reply ONLY on the context verses below. Do not invent verses.\n\n"

    "Structure your reply exactly like this:\n\n"
    "✨ **My Words to You:**\n[Write exactly ONE short, comforting quote adapted from the context verses here.]\n\n"
    "📖 **The Eternal Truth:**\n[Write exactly TWO short sentences breaking down the root cause of their pain.]\n\n"
    "🛠️ **Your Path Forward:**\n[Give exactly ONE simple, physical, micro-action they can do right this second.]\n\n"
    "Context:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

# 4. Handle chat query
user_query = st.text_input("Look within— what is the real battle you are fighting today, Arjun?")

if user_query:
    if not GROQ_API_KEY:
        st.error("🔒 Configuration Error: 'GROQ_API_KEY' is missing. Add it to .env or Streamlit Secrets.")
    else:
        with st.spinner("HARE KRISHNA HARE KRISHNA KRISHNA KRISHNA HARE HARE..."):
            try:
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=MODEL_NAME,
                    temperature=0.4,
                )

                # Retrieve ONCE, so the verses shown are exactly the verses the model saw
                retrieved_docs = retriever.invoke(user_query)

                if not retrieved_docs:
                    st.warning("No matching verses were found. Try describing your situation in a few more words.")
                    st.stop()

                chain = prompt_template | llm | StrOutputParser()
                response_text = chain.invoke({
                    "context": format_docs(retrieved_docs),
                    "input": user_query,
                })

                st.markdown("### 🏹 Words from Achyuta")
                st.info(response_text)

                st.write("---")
                with st.expander("🔬 View Source Verses Used by Madhav"):
                    for idx, doc in enumerate(retrieved_docs):
                        chapter_val = doc.metadata.get("chapter", "Unknown")
                        verse_val = doc.metadata.get("verse", "Unknown")
                        theme_val = doc.metadata.get("theme", "General Guidance")

                        st.markdown(
                            f"🌟 **Reference #{idx+1}: Chapter {chapter_val}, Verse {verse_val}** | *Theme: {theme_val}*"
                        )
                        st.info(doc.page_content)
                        st.write("")

            except Exception as e:
                st.error(f"An error occurred: {e}")
