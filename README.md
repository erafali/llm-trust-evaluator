# 🧠 LLM Trustworthiness Evaluator

A research-oriented system to evaluate how different LLM agents behave across models using multiple trust metrics.

---

## 🚀 Features

- 🔹 Multi-Agent System (basic, reasoning, concise, etc.)
- 🔹 Multi-Model Support (Phi-3, Mistral via Ollama)
- 🔹 Trust Metrics:
  - Consistency
  - Hallucination Detection
  - LLM-as-Judge (Correctness, Reasoning, Clarity, Safety)
- 🔹 Final Weighted Scoring System
- 🔹 Interactive UI (Streamlit)
- 🔹 Visualizations:
  - Bar Charts
  - Radar Chart
  - Leaderboard

---

## 🧠 Objective

To evaluate the **trustworthiness and robustness of LLM agents** across different prompting strategies and models.

---

## ⚙️ Tech Stack

- Python
- Streamlit
- Ollama (Local LLMs)
- Pandas
- NumPy
- Matplotlib

---

### Install Ollama

Download and install from:
https://ollama.com

Then run:

ollama pull phi3
ollama pull mistral

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Example Insights

- Concise agents show higher consistency  
- Reasoning agents provide better explanation quality  
- Trade-off exists between clarity and stability  

---

## ⚠️ Note

This project uses **local LLMs (Ollama)**.  
For deployment, API-based models (OpenAI, Gemini) can be integrated.

---

## 📌 Future Work

- Add more models (API-based)
- Improve hallucination detection (semantic/RAG)
- Deploy via backend API

---

## 👨‍💻 Author

**Eraf Ali**