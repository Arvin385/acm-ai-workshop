import os
import requests
from bs4 import BeautifulSoup
import urllib
import time
from urllib.parse import urlparse, parse_qs, unquote
from rich.console import Console
from rich.prompt import Prompt

# -----------------------
# LangChain FAISS imports
# -----------------------
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------
# Load lightweight sentence-transformers model
# -----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------
# Gemini LLM Setup
# -----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def query_gemini(prompt_text):
    """Send a prompt to Gemini 2.5 Pro and return the generated text."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    data = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt_text}]}
        ]
    }

    response = requests.post(BASE_URL, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()

    text_output = ""
    if "candidates" in result and len(result["candidates"]) > 0:
        for part in result["candidates"][0]["content"]["parts"]:
            text_output += part.get("text", "")
    return text_output

# -----------------------
# Keyword Detection
# -----------------------
KEYWORDS = ["stock", "news", "weather", "when", "time", "crypto"]

def needs_timely_info(user_input):
    return any(keyword.lower() in user_input.lower() for keyword in KEYWORDS)

# -----------------------
# Web Scraping Functions
# -----------------------
def extract_real_url(duck_url):
    parsed = urlparse(duck_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path == "/l/":
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return duck_url

def fetch_web_info(query, max_links=5, max_paragraphs=7):
    """Fetch text from the first few paragraphs of top DuckDuckGo results."""
    query_encoded = urllib.parse.quote_plus(query)
    search_url = f"https://duckduckgo.com/html/?q={query_encoded}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Failed to fetch search results: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    # NEW FIX: select links the same way as the working web_LLM.py
    links = []
    for a_tag in soup.find_all("a", class_="result__a")[:max_links]:
        href = a_tag.get("href")
        if href:
            real_url = extract_real_url(href)
            links.append(real_url)

    if not links:
        return "No search links found."

    # Collect text from links
    collected_text = []
    for link in links:
        try:
            resp = requests.get(link, headers=headers, timeout=10)
            resp.raise_for_status()
            page_soup = BeautifulSoup(resp.text, "html.parser")
            texts = [tag.get_text(strip=True) for tag in page_soup.find_all(["p","div","span"])[:max_paragraphs]]
            collected_text.extend(texts)
        except Exception:
            continue

    if not collected_text:
        return "No timely info found."
    return " ".join(collected_text)

# -----------------------
# Rich Console
# -----------------------
console = Console()

# -----------------------
# FAISS Memory Setup
# -----------------------
VECTOR_DB_PATH = "faiss_index"

if os.path.exists(VECTOR_DB_PATH):
    faiss = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    faiss = None

# -----------------------
# Main Chat Loop
# -----------------------
def chat_loop():
    global faiss
    console.print("[bold green]Gemini RAG Chat[/bold green]. Type 'exit' to quit.")
    conversation_history = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if user_input.lower() in ("exit", "quit"):
            break

        # Web scraping for timely info
        if needs_timely_info(user_input):
            with console.status("[bold yellow]Fetching timely info from the web...[/bold yellow]", spinner="dots"):
                web_info = fetch_web_info(user_input)
                time.sleep(0.5)
            if web_info not in ("No search links found.", "No timely info found."):
                prompt_with_web = f"{user_input}\n\nUse this up-to-date information to answer:\n{web_info}"
            else:
                prompt_with_web = user_input
        else:
            prompt_with_web = user_input

        # FAISS retrieval
        retrieved_text = ""
        if faiss:
            relevant_docs = faiss.similarity_search(user_input, k=3)
            retrieved_text = " ".join([doc.page_content for doc in relevant_docs])

        # Final prompt for Gemini
        llm_prompt = prompt_with_web
        if retrieved_text:
            llm_prompt += f"\n\nConsider this past info:\n{retrieved_text}"

        # Query Gemini
        with console.status("[bold yellow]LLM is generating...[/bold yellow]", spinner="dots"):
            try:
                response = query_gemini(llm_prompt)
            except requests.exceptions.HTTPError as e:
                response = f"[Error] Gemini API request failed: {e}"

        # Update FAISS memory
        chunks = text_splitter.split_text(user_input + " " + response)
        if not faiss:
            faiss = FAISS.from_texts(chunks, embeddings)
        else:
            faiss.add_texts(chunks)

        faiss.save_local(VECTOR_DB_PATH)

        # Print chat
        console.print("[bold magenta]Gemini:[/bold magenta]", response)
        conversation_history.append({"user": user_input, "gemini": response})

if __name__ == "__main__":
    chat_loop()
