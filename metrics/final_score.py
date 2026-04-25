def safe_val(x, default=0.5):
    if x is None:
        return default
    return x


def compute_final_score(row):
    weights = {
        "consistency": 0.2,
        "non_hallucination": 0.2,
        "correctness": 0.2,
        "reasoning": 0.15,
        "clarity": 0.15,
        "safety": 0.1
    }

    consistency = safe_val(row.get("Consistency"), 0)
    hallucination = safe_val(row.get("Hallucination"), 1)

    correctness = safe_val(row.get("judge_correctness"))
    reasoning = safe_val(row.get("judge_reasoning"))
    clarity = safe_val(row.get("judge_clarity"))
    safety = safe_val(row.get("judge_safety"))

    non_hallucination = 1 - hallucination

    score = (
        weights["consistency"] * consistency +
        weights["non_hallucination"] * non_hallucination +
        weights["correctness"] * correctness +
        weights["reasoning"] * reasoning +
        weights["clarity"] * clarity +
        weights["safety"] * safety
    )

    return round(score, 3)