def build_prompt(prompt, agent_type):

    if agent_type == "basic":
        return prompt

    elif agent_type == "strict":
        return f"""
        Answer strictly.
        If you are unsure, say 'I don't know'.
        Do not guess.

        Question: {prompt}
        """

    elif agent_type == "reasoning":
        return f"""
        Solve step-by-step logically.
        Then give final answer.

        Question: {prompt}
        """

    elif agent_type == "concise":
        return f"""
        Answer in ONE short sentence only.

        Question: {prompt}
        """

    elif agent_type == "math":
        return f"""
        Solve the problem step-by-step.
        Give final numeric answer only.

        Question: {prompt}
        """

    elif agent_type == "factual":
        return f"""
        Answer factually correct.
        If unsure, say 'I don't know'.

        Question: {prompt}
        """
    return prompt