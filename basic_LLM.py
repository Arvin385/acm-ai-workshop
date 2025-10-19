import os
import requests
import json

# Make sure your API key is set in environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-pro"  # the model you want to use

BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def query_gemini(prompt_text):
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
    response.raise_for_status()  # will raise an error if something goes wrong

    result = response.json()
    # Extract the text output from the model
    text_output = ""
    if "candidates" in result and len(result["candidates"]) > 0:
        for part in result["candidates"][0]["content"]["parts"]:
            text_output += part.get("text", "")
    return text_output

if __name__ == "__main__":
    print("Connected to Gemini 2.5 Pro. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            output = query_gemini(user_input)
            print("Gemini:", output)
        except requests.HTTPError as e:
            print("HTTP error:", e)
        except Exception as e:
            print("Error:", e)
