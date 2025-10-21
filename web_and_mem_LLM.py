import os
import requests
from bs4 import BeautifulSoup
import urllib
import time

from urllib.parse import urlparse, parse_qs, unquote
from rich.console import Console
from rich.spinner import Spinner
from rich.prompt import Prompt
from rich.text import Text

# -----------------------
# LangChain FAISS imports
# -----------------------
from langchain.embeddings import LlamaCppEmbeddings  # or your embedding
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# load lightweight sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------
# Gemini LLM Setup
# -----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def query_gemini(prompt_text):
    """Sends a prompt to Gemini 2.5 Pro and returns the generated text."""
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
    """Checks if the input contains keywords that indicate RAG is needed."""
    return any(keyword.lower() in user_input.lower() for keyword in KEYWORDS)

# -----------------------
# Web Scraping Function
# -----------------------
def extract_real_url(duck_url):
    parsed = urlparse(duck_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path == "/l/":
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return duck_url

def fetch_web_info(query, max_links=5, max_paragraphs=7):
    """Fetches text from the first few paragraphs of the top DuckDuckGo results."""
    query_encoded = urllib.parse.quote_plus(query)
    search_url = f"https://duckduckgo.com/html/?q={query_encoded}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(search_url, headers=headers, timeout=5)
        response.raise_for_status()
    except Exception as e:
        return f"Failed to fetch search results: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    # Try multiple ways to find result links
    links = []
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href")
        if href and ("http" in href or "duckduckgo.com/l/?" in href):
            links.append(extract_real_url(href))
        if len(links) >= max_links:
            break

    if not links:
        return "No search links found."

    collected_text = []
    for link in links:
        try:
            resp = requests.get(link, headers=headers, timeout=5)
            resp.raise_for_status()
            page_soup = BeautifulSoup(resp.text, "html.parser")

            # Get text from <p> and fallback to <div>
            texts = [p.get_text(strip=True) for p in page_soup.find_all("p")[:max_paragraphs]]
            if not texts:
                texts = [div.get_text(strip=True) for div in page_soup.find_all("div")[:max_paragraphs]]
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

    # Text splitter and embeddings for FAISS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if user_input.lower() in ("exit", "quit"):
            break

        # Check if we need to scrape timely info
        if needs_timely_info(user_input):
            with console.status("[bold yellow]Fetching timely info from the web...[/bold yellow]", spinner="dots"):
                web_info = fetch_web_info(user_input)
                time.sleep(0.5)  # spinner visible
            prompt_with_web = f"{user_input}\n\nUse this up-to-date information to answer:\n{web_info}"
        else:
            prompt_with_web = user_input

        # FAISS memory retrieval
        retrieved_text = ""
        if faiss:
            relevant_docs = faiss.similarity_search(user_input, k=3)
            retrieved_text = " ".join([doc.page_content for doc in relevant_docs])

        # Build final prompt for Gemini
        llm_prompt = prompt_with_web
        if retrieved_text:
            llm_prompt += f"\n\nConsider this past info:\n{retrieved_text}"

        # Call Gemini
        with console.status("[bold yellow]LLM is generating...[/bold yellow]", spinner="dots"):
            response = query_gemini(llm_prompt)

        # Store new input + response in FAISS
        chunks = text_splitter.split_text(user_input + " " + response)
        if not faiss:
            faiss = FAISS.from_texts(chunks, embeddings)
        else:
            faiss.add_texts(chunks)

        # Save FAISS index safely
        faiss.save_local(VECTOR_DB_PATH)

        # Print chat
        console.print("[bold magenta]Gemini:[/bold magenta]", response)
        conversation_history.append({"user": user_input, "gemini": response})


if __name__ == "__main__":
    chat_loop()
