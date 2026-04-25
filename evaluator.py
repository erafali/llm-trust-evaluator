from llm import call_llm
from metrics.consistency import consistency_score
from metrics.hallucination import hallucination_check
from metrics.trust_score import compute_trust
from agents import build_prompt
import streamlit as st


@st.cache_data(show_spinner=False)
def evaluate_prompt(prompt, model, agent_type, use_llm_hallucination=False):
    
    responses = []
    hallucinations = []

    for _ in range(2):

        agent_prompt = build_prompt(prompt, agent_type)
        res = call_llm(agent_prompt, model)

        if "Error" in res:
            continue

        responses.append(res)
        hallucinations.append(
            hallucination_check(
                res,
                prompt,
                responses=responses,
                use_llm=use_llm_hallucination
            )
        )
    
    
    if len(responses) < 2:
        return {
                "model": model,
                "agent": agent_type,
                "prompt": prompt,
                "responses": responses,
                "consistency": 0,
                "hallucination": 1,
                "trust": 0
            }
    


    consistency = consistency_score(responses)
    hallucination_score = sum(hallucinations) / len(hallucinations)
    trust_score = compute_trust(consistency, hallucination_score)

    
    
    return {
            "model": model,
            "agent": agent_type,
            "prompt": prompt,
            "responses": responses,
            "consistency": consistency,
            "hallucination": hallucination_score,
            "trust": trust_score
        }