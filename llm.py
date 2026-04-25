import requests

def call_llm(prompt, model="phi3"):

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 50
                }
            }
        )

        # # 🔍 DEBUG PRINT
        # print("RAW RESPONSE:", response.text)

        data = response.json()

        return data.get("response", "No response")

    except Exception as e:
        return f"ERROR: {str(e)}"