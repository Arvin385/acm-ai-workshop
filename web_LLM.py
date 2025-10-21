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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# -----------------------
# Gemini Query Function
# -----------------------
def query_gemini(prompt_text):
    """
    Sends a prompt to Gemini 2.5 Pro and returns the generated text.
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    data = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
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
KEYWORDS = ["stock", "news", "weather", "when", "time", "crypto"] ## ADD YOUR OWN KEYWORD DETECTION FOR WEBSCRAPING

def needs_timely_info(user_input):
    """
    Checks if the input contains keywords that indicate RAG is needed.
    """
    return any(keyword.lower() in user_input.lower() for keyword in KEYWORDS)

# -----------------------
# Web Scraping Function
# -----------------------
import requests
from bs4 import BeautifulSoup
import urllib
from urllib.parse import urlparse, parse_qs, unquote

def extract_real_url(duck_url):
    """
    Converts DuckDuckGo redirect link to the actual target URL.
    """
    parsed = urlparse(duck_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path == "/l/":
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return duck_url

def fetch_web_info(query, max_links=3, max_paragraphs=5):
    """
    Fetches text from the first few paragraphs of the top N DuckDuckGo results.
    """
    query_encoded = urllib.parse.quote_plus(query)
    search_url = f"https://duckduckgo.com/html/?q={query_encoded}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    # Get DuckDuckGo search results
    response = requests.get(search_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract top result links
    links = []
    for a_tag in soup.find_all("a", class_="result__a")[:max_links]:
        href = a_tag.get("href")
        if href:
            real_url = extract_real_url(href)
            links.append(real_url)

    # Scrape each link for text
    collected_text = []
    for link in links:
        try:
            resp = requests.get(link, headers=headers, timeout=5)
            resp.raise_for_status()
            page_soup = BeautifulSoup(resp.text, "html.parser")

            # Get text from <p> and <div> tags
            texts = [p.get_text(strip=True) for p in page_soup.find_all("p")[:max_paragraphs]]
            if not texts:  # fallback to <div> if <p> is empty
                texts = [div.get_text(strip=True) for div in page_soup.find_all("div")[:max_paragraphs]]
            collected_text.extend(texts)
        except Exception as e:
            # Skip links that fail
            continue

    return " ".join(collected_text) or "No timely info found."

console = Console()

# -----------------------
# Main Chat Function
# -----------------------
def chat_loop():
    console.print("[bold green]Gemini RAG Chat[/bold green]. Type 'exit' to quit.")
    conversation_history = []

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if user_input.lower() in ("exit", "quit"):
            break

        # Check if we need to scrape
        if needs_timely_info(user_input):
            with console.status("[bold yellow]Fetching timely info from the web...[/bold yellow]", spinner="dots"):
                web_info = fetch_web_info(user_input)
                time.sleep(0.5)  # small delay to make spinner visible
            with console.status("[bold yellow]LLM is generating...[/bold yellow]", spinner="dots"):
                final_prompt = f"{user_input}\n\nUse this up-to-date information to answer:\n{web_info}"
                response = query_gemini(final_prompt)
        else:
            response = query_gemini(user_input)

        console.print("[bold cyan]You:[/bold cyan]", user_input)
        console.print("[bold magenta]Gemini:[/bold magenta]", response)
        conversation_history.append({"user": user_input, "gemini": response})

if __name__ == "__main__":
    chat_loop()