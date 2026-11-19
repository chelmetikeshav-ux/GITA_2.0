import os
import json
import glob
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

INPUT_FOLDER = "./gita_chapters"
DB_PATH = "./gita_complete_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    """Read every chapter JSON and return a list of LangChain Documents."""
    json_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.json")))
    documents = []

    for file_path in json_files:
        filename = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)

        verses_list = (
            chapter_data if isinstance(chapter_data, list) else chapter_data.get("verses", [])
        )

        for entry in verses_list:
            v_num = entry.get("verse", "Unknown")
            c_num = entry.get("chapter", "Unknown")

            translation = (entry.get("translation") or "").strip()
            explanation = (entry.get("simple_explanation") or "").strip()
            life_app = (entry.get("life_application") or "").strip()
            keywords_list = entry.get("keywords") or []
            theme = entry.get("theme", "General")

            if not translation:
                print(f"Skipping {filename} Verse {v_num}: no translation")
                continue

            # Only include fields that actually exist, so empty placeholders
            # don't dilute the embeddings.
            parts = [f"Chapter {c_num}, Verse {v_num} ({theme})", f"Translation: {translation}"]
            if explanation:
                parts.append(f"Explanation: {explanation}")
            if life_app:
                parts.append(f"Life Application: {life_app}")
            if isinstance(keywords_list, list) and keywords_list:
                parts.append(f"Keywords: {', '.join(keywords_list)}")
            page_content = "\n".join(parts)

            metadata = {
                "id": str(entry.get("id", f"{c_num}_{v_num}")),
                "chapter": int(c_num) if str(c_num).isdigit() else 0,
                "verse": int(v_num) if str(v_num).isdigit() else 0,
                "theme": str(theme),
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def build_vector_db(embedding_model=None):
    """Wipe any old database and build a fresh one. Returns the number of verses stored."""
    documents = load_documents()

    if not documents:
        raise FileNotFoundError(
            f"No verses found. Make sure '{INPUT_FOLDER}' contains your chapter .json files."
        )

    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Clear any old data so we never end up with duplicates. Clearing through Chroma
    # (instead of deleting the folder) is safe even if the database is already open
    # in this process, e.g. when the Streamlit app triggers a rebuild.
    if os.path.exists(DB_PATH):
        try:
            Chroma(persist_directory=DB_PATH, embedding_function=embedding_model).delete_collection()
        except Exception:
            # Unreadable or incompatible database: remove the folder and start over
            shutil.rmtree(DB_PATH, ignore_errors=True)

    Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=DB_PATH,
    )
    return len(documents)


if __name__ == "__main__":
    print("Building vector database... This might take a few moments.")
    count = build_vector_db()
    print(f"Success! {count} verses saved to '{DB_PATH}'")
