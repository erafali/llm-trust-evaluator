import json
from evaluator import evaluate_prompt

if __name__ == "__main__":
    with open("data/prompts.json") as f:
        prompts = json.load(f)

    for item in prompts:
        question = item["question"]
        category = item["category"]

        print(f"\n==============================")
        print(f"Category: {category}")
        print("Question:", question)

        for agent in ["basic", "strict", "reasoning", "concise", "math", "factual"]:
            for model in ["phi3", "mistral"]:
                
                result = evaluate_prompt(question, model, agent)

                print(f"\n--- Agent: {agent} | Model: {model} ---")

                print("Responses:")
                for r in result["responses"]:
                    print("-", r)

                print("Consistency:", round(result["consistency"], 3))
                print("Hallucination:", result["hallucination"])
                print("Trust Score:", round(result["trust"], 3))