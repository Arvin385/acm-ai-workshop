# acm-ai-workshop
Fall 2025 Workshop on AI for ACM. There are multiple programs in this repo. The first is a basic API for a Gemini LLM. Students will note two major difficulties when using the LLM: 1. The pre-trained model often times gives out-of-date information. 2. Its "memory" is limited to the immediately stated prompt. The others correct for these issues.

# Fall 2025 ACM Workshop: AI with Gemini and Memory

Welcome to the **Fall 2025 ACM Workshop on AI**! Most people use AI tools like ChatGPT or Gemini, but those tools are built on top of a basic LLM. A basic LLM can only respond using what it learned during training — it can’t look up new information or remember past conversations.
In this workshop, we’ll improve the basic Gemini LLM by adding two key features: web retrieval for up-to-date answers and FAISS memory so it can remember previous chats.

This repository contains **three programs**:

1. **Basic Gemini LLM API** – A simple LLM interface.
2. **Web + Gemini LLM** – Adds web scraping to provide up-to-date info.
3. **Web + Memory + Gemini LLM** – Adds a FAISS memory system to remember prior interactions.

---

## 1. Prerequisites

### 1.1 Gemini API Key

Before running any code, you need a **Gemini API key**:

1. Go to [Google AI Studio](https://aistudio.google.com/app/api-keys).
2. Create a project and generate a **Gemini API key**.
3. Store the key as an environment variable:

Linux/macOS:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Windows: (make sure to use quotes around the api key)
```bash
$env:GEMINI_API_KEY="your_api_key_here"
```
---

### 1.2 Create a Python Virtual Environment

We recommend using a virtual environment to manage dependencies:
 (Note: We reccomend using Python 3.12 for this workshop as it has better compatibility)

```bash
# If you have version 3.12 by default, run:
python -m venv acm-ai-venv
# RECCOMENDED: 
py -3.12 -m venv acm-ai-venv
source acm-ai-venv/bin/activate      # On Linux/macOS

acm-ai-venv\Scripts\activate         # On Windows
```
On Windows, if you encounter an execution policy error, you may have to run the following in Command Prompt:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser   
```

---

### 1.3 Install Dependencies
If you haven't already, you need to install Python. 
To check if you have it or not, run:
```bash
python --version
```
If you get an actual version, you are set. If you get some error about Python not existing, then you need to install it (Python website for Windows, and use Homebrew for Mac). Again, we reccomend installing Python 3.12 for this workshop.

```bash
pip install -r requirements.txt
```

---

## 2. Running the Programs

### 2.1 Basic Gemini LLM

Run the simplest LLM API:

```bash
python basic_LLM.py
```

* Type a query and see Gemini respond.
* **Observations:**

  1. The model may give **out-of-date information**.
  2. It **forgets context** beyond the immediate prompt.

---

### 2.2 Gemini + Web Scraping

Run:

```bash
python web_LLM.py
```

* Fetches **timely information** from the web.
* **Customizable parameters:**

#### Keywords (Lines ~53–57)

```python
KEYWORDS = ["stock", "news", "weather", "when", "time", "crypto"]
```

* Add or remove keywords to control **when web scraping happens**. Otherwise, webscraping will be bypassed. 

#### Web scraping depth (Lines ~60–80)

```python
def fetch_web_info(query, max_links=3, max_paragraphs=5):
```

* Adjust `max_links` or `max_paragraphs` to balance **response quality vs speed**.

---

### 2.3 Gemini + Web + Memory (FAISS)

Run:

```bash
python web_and_mem_LLM.py
```

* Stores past interactions in a **FAISS vector store** for persistent memory.
* **Customizable parameters:**

#### 1. FAISS chunk size and overlap (Lines ~138–145)

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

* `chunk_size`: Larger → fewer docs, faster retrieval, less granular memory.
* `chunk_overlap`: Helps maintain context across chunks.

#### 2. Embedding model (Line ~28)

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

* Switch to a **larger model** for better semantic understanding, or **smaller** for faster responses on weaker machines.

#### 3. Number of FAISS documents retrieved (Line ~140)

```python
relevant_docs = faiss.similarity_search(user_input, k=3)
```

* Increase `k` to consider more memory context.
* Decrease `k` for faster responses and simpler prompts.

#### 4. Web + Memory prompt formatting (Lines ~122–130)

```python
llm_prompt = prompt_with_web
if retrieved_text:
    llm_prompt += f"\n\nConsider this past info:\n{retrieved_text}"
```

* Customize to **prioritize web info vs memory** or format prompts differently.

---

## 3. Hands-On Exercises

1. **Add a keyword**: Try `"sports"` in `KEYWORDS` and ask Gemini for scores.
2. **Increase scraping depth**: Set `max_links=5` and `max_paragraphs=10` and observe the effect.
3. **Adjust memory granularity**: Change `chunk_size` to 200 and `chunk_overlap` to 50, then ask about past interactions.
4. **Switch embeddings**: Try `"all-MiniLM-L12-v2"` for more semantic accuracy.

---

## 4. Notes and Tips

* Gemini **requires an internet connection** for API calls.
* FAISS memory is **local**, but no LLM model needs to run on your machine.
* Adjust **keywords, chunk size, and k** to balance **speed vs context richness**.
* When web-retrieving, it is more accurate to send in the user's query to an LLM to determine whether a web search is even required. This would replace the keyword comparison, heavily trading content accuracy over time efficiency. 
---

## 5. Summary

By the end of this workshop, students will be able to:

* Query a **Gemini LLM**.
* Fetch **real-time web information**.
* Build a **persistent memory** with FAISS.
* **Customize prompts, embeddings, memory, and web scraping** for personalized AI assistants.
