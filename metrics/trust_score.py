def compute_trust(consistency, hallucination):
    return 0.7 * consistency + 0.3 * (1 - hallucination)