from difflib import SequenceMatcher

def consistency_score(responses):
    responses = [r.strip().lower() for r in responses]
    scores = []

    for i in range(len(responses) - 1):
        score = SequenceMatcher(None, responses[i], responses[i+1]).ratio()
        scores.append(score)
    
    if len(scores) == 0:
        return 0

    return sum(scores) / len(scores)