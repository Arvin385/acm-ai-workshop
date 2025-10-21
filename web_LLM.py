import os
import requests
from bs4 import BeautifulSoup
import urllib
from urllib.parse import urlparse, parse_qs, unquote
import time
from rich.console import Console
from rich.prompt import Prompt

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

console = Console()

# -----------------------
# Gemini Query Function
# -----------------------
def query_gemini(prompt_text):
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

def fetch_web_info(query, max_links=5, max_paragraphs=10):
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

    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "duckduckgo.com/l/" in href or (href.startswith("http") and "duckduckgo.com" not in href):
            real_url = extract_real_url(href)
            if real_url not in links:
                links.append(real_url)
        if len(links) >= max_links:
            break

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

    return " ".join(collected_text) if collected_text else "No timely info found."

# -----------------------
# Main Chat Loop
# -----------------------
def chat_loop():
    console.print("[bold green]Gemini RAG Chat[/bold green]. Type 'exit' to quit.")

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if user_input.lower() in ("exit", "quit"):
            break

        final_prompt = user_input

        if needs_timely_info(user_input):
            with console.status("[bold yellow]Fetching timely info from the web...[/bold yellow]", spinner="dots"):
                web_info = fetch_web_info(user_input)
                time.sleep(0.5)
            if web_info != "No timely info found.":
                final_prompt += f"\n\nUse this up-to-date information to answer:\n{web_info}"

        with console.status("[bold yellow]LLM is generating...[/bold yellow]", spinner="dots"):
            try:
                response = query_gemini(final_prompt)
            except requests.exceptions.HTTPError as e:
                response = f"[Error] Gemini API request failed: {e}"

        console.print("[bold cyan]You:[/bold cyan]", user_input)
        console.print("[bold magenta]Gemini:[/bold magenta]", response)

if __name__ == "__main__":
    chat_loop()
