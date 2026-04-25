import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from evaluator import evaluate_prompt
from judge import llm_judge
from metrics.final_score import compute_final_score

st.set_page_config(layout="wide")
st.title("🧠 LLM Trustworthiness Evaluator")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Settings")

questions_input = st.sidebar.text_area(
    "Enter multiple questions (one per line)",
    "What is 2+2?\nIf A > B and B > C, who is largest?"
)

questions = [q.strip() for q in questions_input.split("\n") if q.strip()]

models = st.sidebar.multiselect(
    "Select Models",
    ["phi3", "mistral"],
    default=["phi3"]
)

agents = st.sidebar.multiselect(
    "Select Agents",
    ["basic", "strict", "reasoning", "concise", "math", "factual"],
    default=["basic", "reasoning", "concise"]
)

use_judge = st.sidebar.checkbox("Use LLM-as-Judge (slower)", value=True)
use_llm_hallucination = st.sidebar.checkbox("Use LLM Hallucination Check", value=False)

run = st.sidebar.button("🚀 Run Evaluation")

if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.success("Cache Cleared!")

# ---------------- PROCESS ---------------- #
if run and questions:

    @st.cache_data(show_spinner=False)
    def run_full_evaluation(questions, agents, models, use_judge,use_llm_hallucination):
        rows = []

        for q in questions:
            for agent in agents:
                for model in models:

                    res = evaluate_prompt(q, model, agent, use_llm_hallucination)

                    judge_scores = {
                        "judge_correctness": None,
                        "judge_reasoning": None,
                        "judge_clarity": None,
                        "judge_safety": None,
                    }

                    if use_judge and res["responses"]:
                        judge_scores = llm_judge(q, res["responses"][0], model)

                    rows.append({
                        "Question": q,
                        "Agent": agent,
                        "Model": model,
                        "Consistency": res["consistency"],
                        "Hallucination": res["hallucination"],
                        "Trust": res["trust"],
                        **judge_scores,
                        "Responses": res["responses"]
                    })

        return pd.DataFrame(rows)

    with st.spinner("🔄 Running evaluation... Please wait"):
        df = run_full_evaluation(
            questions, agents, models, use_judge, use_llm_hallucination
        )


    df["FinalScore"] = df.apply(compute_final_score, axis=1)

    # ---------------- TABS ---------------- #
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧾 Responses",
        "📊 Table",
        "📈 Charts",
        "🏆 Leaderboard"
    ])
    
    # -------- TAB 1: Responses -------- #
    with tab1:
        st.subheader("Agent Responses")
        for _, r in df.iterrows():
            with st.expander(f"{r['Question']} | {r['Agent']} | {r['Model']}"):
                st.write(r["Responses"][0])
                with st.expander("🔍 Show consistency responses"):
                    for resp in r["Responses"]:
                        st.write(resp)
    # st.markdown("---")
    # -------- TAB 2: Table -------- #
    with tab2:
        st.subheader("Evaluation Metrics")
        show_cols = [c for c in df.columns if c != "Responses"]
        st.dataframe(df[show_cols])
        st.markdown("---")

        csv = df[show_cols].to_csv(index=False)

        st.download_button(
            "📥 Download Results (CSV)",
            data=csv,
            file_name="llm_eval_results.csv",
            mime="text/csv"
        )

    
    # -------- TAB 3: Charts -------- #
    with tab3:
        st.subheader("Trust Comparison")

        chart_df = df.copy()
        chart_df["Agent_Model"] = chart_df["Agent"] + " | " + chart_df["Model"]
        st.bar_chart(chart_df.set_index("Agent_Model")["FinalScore"])

        
        st.subheader("Consistency vs Hallucination")
        
        chart_df = df.copy()
        chart_df["Agent_Model"] = chart_df["Agent"] + " | " + chart_df["Model"]
        st.bar_chart(
            chart_df.set_index("Agent_Model")[
                ["Consistency", "Hallucination"]
            ]
        )

        # ---- Radar Chart (per Agent average) ---- #
        st.subheader("📊 Radar Chart (Agent Comparison)")

        import numpy as np
        import matplotlib.pyplot as plt

        # ---- 1) Build metrics list (clean + no duplicates) ---- #
        metrics = ["Consistency", "FinalScore"]  # use FinalScore instead of Trust


        # Add judge metrics only if present
        judge_cols = [
            "judge_correctness",
            "judge_reasoning",
            "judge_clarity",
            "judge_safety"
        ]
        if use_judge:
            metrics += [c for c in judge_cols if c in df.columns]

        # Invert hallucination (lower is better → higher is better)
        if "Hallucination" in df.columns:
            df["NonHallucination"] = 1 - df["Hallucination"]
            metrics.append("NonHallucination")
       
        # Ensure unique + existing columns only
        metrics = [m for i, m in enumerate(metrics) if m in df.columns and m not in metrics[:i]]

        # ---- 2) Aggregate per agent ---- #
        agg = df.groupby("Agent")[metrics].mean().fillna(0)

        # ---- 3) Normalize to [0,1] (important for clean radar) ---- #
        norm = (agg - agg.min()) / (agg.max() - agg.min() + 1e-9)

        # radar plot
        labels = list(norm.columns)
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        # ---- 4) Plot ---- #
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))


        for agent in norm.index:
            values = norm.loc[agent].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=agent)
            ax.fill(angles, values, alpha=0.1)

        # ---- 5) Clean labels & layout ---- #
        pretty_labels = [l.replace("_", " ").title() for l in labels]

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(pretty_labels, fontsize=10)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

        ax.set_title("Agent Comparison (Normalized Metrics)", pad=20, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        plt.tight_layout()

        st.pyplot(fig)
    # st.markdown("---")
    # -------- TAB 4: Leaderboard -------- #
    with tab4:
        st.subheader("🏆 Leaderboard (Final Score)")

        # Aggregate using FINAL SCORE
        board = df.groupby(["Agent", "Model"]).agg({
            "FinalScore": "mean",
            "Consistency": "mean",
            "Hallucination": "mean"
        }).reset_index()

        # Sort by Final Score
        board = board.sort_values(by="FinalScore", ascending=False)

        st.dataframe(board, use_container_width=True)

        # 🏆 Best Performer (based on FinalScore)
        top = board.iloc[0]

        st.success(f"""
        🏆 Best Performer  
        Agent: {top['Agent']}  
        Model: {top['Model']}  
        Final Score: {top['FinalScore']:.3f}
        """)

        csv_board = board.to_csv(index=False)

        st.download_button(
            "📥 Download Leaderboard",
            data=csv_board,
            file_name="leaderboard.csv",
            mime="text/csv"
        )