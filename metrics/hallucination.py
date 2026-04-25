from difflib import SequenceMatcher
from llm import call_llm


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def rule_based_check(response, prompt):
    response_lower = response.lower()
    prompt_lower = prompt.lower()

    if "capital of france" in prompt_lower:
        return 0 if "paris" in response_lower else 1

    if "2+2" in prompt_lower:
        return 0 if ("4" in response_lower or "four" in response_lower) else 1

    if "mars" in prompt_lower or "sun" in prompt_lower:
        if any(word in response_lower for word in ["no", "not", "does not"]):
            return 0
        return 1

    return 0  # unknown → assume no hallucination


def consistency_based_check(responses):
    if len(responses) < 2:
        return 0

    sim = similarity(responses[0], responses[1])

    # low similarity → possible hallucination
    if sim < 0.3:
        return 1
    return 0


def llm_based_check(response, prompt, model="phi3"):
    eval_prompt = f"""
Check if the response contains hallucinated or incorrect information.

Answer ONLY:
Hallucination: Yes or No

Question: {prompt}
Response: {response}
"""

    try:
        res = call_llm(eval_prompt, model).lower()
        if "yes" in res:
            return 1
        return 0
    except:
        return 0


def hallucination_check(response, prompt, responses=None, use_llm=False):
    """
    Final hybrid hallucination score (0 = good, 1 = hallucinated)
    """

    score = 0

    # 1. Rule-based
    score += rule_based_check(response, prompt)

    # 2. Consistency-based
    if responses:
        score += consistency_based_check(responses)

    # 3. LLM-based (optional)
    if use_llm:
        score += llm_based_check(response, prompt)

    # normalize
    return min(score / 3, 1)