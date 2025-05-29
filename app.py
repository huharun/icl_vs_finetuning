import streamlit as st
import ollama
import time
from transformers import pipeline
from prompts import build_prompt
from utils import (
    compute_f1, compute_accuracy, compute_bleu,
    compute_em, compute_rouge, compute_classification_metrics,
    compute_bertscore, compute_chrf
)
import pandas as pd
import altair as alt

label_map = {
    "label_0": "world",
    "label_1": "sports",
    "label_2": "business",
    "label_3": "sci/tech",
    "LABEL_0": "world",
    "LABEL_1": "sports",
    "LABEL_2": "business",
    "LABEL_3": "sci/tech"
}


st.set_page_config(page_title="ICL vs Fine-Tuning", layout="wide")
st.title("📊 In-Context Learning vs Fine-Tuning: Comparative Analysis")

# Sidebar navigation
st.sidebar.title("🔍 Experiment Navigator")

# Clear history button
if st.sidebar.button("🧹 Clear History"):
    st.session_state["comparison_history"] = []

view = st.sidebar.radio("Select View", [
    "Run Comparison",
    "Few-Shot Scaling",
    "Prompt Style Effect",
    "Task-Wise Overview"
])

# ==== RUN COMPARISON ====
if view == "Run Comparison":
    task = st.selectbox("Choose NLP Task:", ["Question Answering (SQuAD)", "Text Classification (AG News)", "Translation (WMT)"])
    question = context = expected = ""

    if task == "Question Answering (SQuAD)":
        question = st.text_area("🔹 Question", "What is Python?")
        context = st.text_area("📘 Context", "Python is a programming language used in AI and web development.")
        expected = st.text_input("✅ Expected Answer", "a programming language")
    elif task == "Text Classification (AG News)":
        context = st.text_area("📰 Headline", "Apple releases new iPhone model")
        expected = st.selectbox("✅ Expected Category", ["World", "Sports", "Business", "Sci/Tech"])
    elif task == "Translation (WMT)":
        context = st.text_area("📝 English Input", "Hello, how are you?")
        expected = st.text_input("✅ Expected French Translation", "Bonjour, comment ça va ?")

    prompt_style = st.selectbox("💬 Prompt Style (for ICL)", ["Zero-Shot", "Few-Shot", "Chain-of-Thought"])

    if st.button("▶️ Run Models"):
        results = []

        # Fine-Tuned Model
        if task == "Question Answering (SQuAD)":
            qa = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
            ft_answer = qa(question=question, context=context)["answer"]
            st.subheader("Fine-Tuned Model Answer")
            st.write(ft_answer)

            results.extend([
                {"Method": "Fine-Tuned BERT", "Metric": "F1", "Score": compute_f1(ft_answer, expected)},
                {"Method": "Fine-Tuned BERT", "Metric": "Exact Match", "Score": compute_em(ft_answer, expected)},
                {"Method": "Fine-Tuned BERT", "Metric": "ROUGE-L", "Score": compute_rouge(ft_answer, expected)}
            ])
        elif task == "Text Classification (AG News)":
            clf = pipeline("text-classification", model="textattack/bert-base-uncased-ag-news")
            raw_label = clf(context)[0]["label"]
            ft_answer = label_map.get(raw_label, raw_label).lower().strip()
            expected = expected.lower().strip()
            st.subheader("Fine-Tuned Model Answer")
            st.write(ft_answer)

            for m, val in compute_classification_metrics(ft_answer, expected).items():
                results.append({"Method": "Fine-Tuned BERT", "Metric": m, "Score": val})
        elif task == "Translation (WMT)":
            translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
            ft_answer = translator(context)[0]["translation_text"]
            st.subheader("Fine-Tuned Model Answer")
            st.write(ft_answer)

            results.extend([
                {"Method": "Fine-Tuned BERT", "Metric": "BLEU", "Score": compute_bleu(ft_answer, expected)},
                {"Method": "Fine-Tuned BERT", "Metric": "BERTScore", "Score": compute_bertscore(ft_answer, expected)},
                {"Method": "Fine-Tuned BERT", "Metric": "ChrF", "Score": compute_chrf(ft_answer, expected)}
            ])

        # ICL Models
        for model in ["llama3", "deepseek-r1"]:
            prompt = build_prompt(context, question, prompt_style, task)
            start = time.time()
            res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            end = time.time()
            icl_answer = res["message"]["content"].strip()

            st.write(f"⏱ Response Time ({model}): {round(end - start, 2)} sec")

            st.subheader(f"ICL Model Answer ({model})")
            st.write(icl_answer)

            if task == "Question Answering (SQuAD)":
                results.extend([
                    {"Method": f"ICL-{model}", "Metric": "F1", "Score": compute_f1(icl_answer, expected)},
                    {"Method": f"ICL-{model}", "Metric": "Exact Match", "Score": compute_em(icl_answer, expected)},
                    {"Method": f"ICL-{model}", "Metric": "ROUGE-L", "Score": compute_rouge(icl_answer, expected)}
                ])
            elif task == "Text Classification (AG News)":
                icl_answer = icl_answer.lower().strip()
                expected = expected.lower().strip()
                for m, val in compute_classification_metrics(icl_answer, expected).items():
                    results.append({"Method": f"ICL-{model}", "Metric": m, "Score": val})
            elif task == "Translation (WMT)":
                results.extend([
                    {"Method": f"ICL-{model}", "Metric": "BLEU", "Score": compute_bleu(icl_answer, expected)},
                    {"Method": f"ICL-{model}", "Metric": "BERTScore", "Score": compute_bertscore(icl_answer, expected)},
                    {"Method": f"ICL-{model}", "Metric": "ChrF", "Score": compute_chrf(icl_answer, expected)}
                ])

        # Store results for Task-Wise Overview
        df = pd.DataFrame(results)
        df["Task"] = task
        if "comparison_history" not in st.session_state:
            st.session_state["comparison_history"] = []
        st.session_state["comparison_history"].append(df)

        st.subheader("📋 Model Performance")
        st.dataframe(df)

        chart = alt.Chart(df).mark_bar().encode(
            x="Method:N",
            y="Score:Q",
            color="Metric:N",
            tooltip=["Score", "Method"]
        ).facet(
            facet=alt.Facet("Metric:N", title=None),
            columns=50
        ).properties(title=f"{task} - Performance Metrics")

        st.altair_chart(chart, use_container_width=True)

# ==== TASK-WISE OVERVIEW ====
elif view == "Task-Wise Overview":
    st.header("📊 Task-Wise Performance Summary")
    if st.session_state.get("comparison_history"):
        all_dfs = pd.concat(st.session_state.comparison_history, ignore_index=True)
        st.dataframe(all_dfs)

        chart = alt.Chart(all_dfs).mark_bar().encode(
            x="Task:N",
            y="Score:Q",
            color="Method:N",
            tooltip=["Task", "Score", "Method"]
        ).facet(
            facet=alt.Facet("Metric:N", title=None),
            columns=50
        ).properties(title="Score Comparison Across Tasks")

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data collected yet. Run comparisons first.")

# ==== FEW-SHOT SCALING ====
elif view == "Few-Shot Scaling":
    st.header("📈 Few-Shot Performance Curve (ICL Only)")
    context = "Python is a widely-used programming language for AI and web development."
    question = "What is Python?"
    expected = "a programming language"
    if st.button("▶️ Run Few-Shot Scaling"):
        with st.spinner("Running few-shot evaluations across models..."):
            records = []
            for n in [0, 1, 3, 5]:
                for model in ["llama3", "deepseek-r1"]:
                    prompt = build_prompt(context, question, "Few-Shot", "Question Answering (SQuAD)", n_shots=n)
                    start = time.time()
                    ans = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])["message"]["content"].strip()
                    end = time.time()

                    st.write(f"Few-Shot {n} → {model}: ⏱ {round(end - start, 2)} sec")
                    f1_score = round(compute_f1(ans, expected), 2)
                    records.append({
                        "Model": model,
                        "Few-Shot Examples": n,
                        "F1": f1_score,
                        "Task": "Question Answering (SQuAD)",
                        "Method": f"ICL-{model}",
                        "Metric": "F1",
                        "Score": f1_score
                    })
        df = pd.DataFrame(records)
        st.dataframe(df)

        if "comparison_history" not in st.session_state:
            st.session_state["comparison_history"] = []
        st.session_state["comparison_history"].append(df)

        st.altair_chart(alt.Chart(df).mark_line(point=True).encode(
            x="Few-Shot Examples:Q",
            y="F1:Q",
            color="Model:N"
        ).properties(title="F1 vs Few-Shot Examples"), use_container_width=True)

# ==== PROMPT STYLE EFFECT ====
elif view == "Prompt Style Effect":
    st.header("🧠 Prompt Style Comparison (QA Only)")
    context = "Python is a widely-used programming language for AI and web development."
    question = "What is Python?"
    expected = "a programming language"
    if st.button("▶️ Compare Styles"):
        with st.spinner("Evaluating prompt styles..."):
            data = []
            for style in ["Zero-Shot", "Few-Shot", "Chain-of-Thought"]:
                for model in ["llama3", "deepseek-r1"]:
                    prompt = build_prompt(context, question, style, "Question Answering (SQuAD)")
                    start = time.time()
                    ans = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])["message"]["content"].strip()
                    end = time.time()

                    st.write(f"{style} / {model} ⏱ Response Time: {round(end - start, 2)} sec")

                    f1_score = round(compute_f1(ans, expected), 2)
                    data.append({
                        "Prompt Style": style,
                        "Model": model,
                        "F1": f1_score,
                        "Task": "Question Answering (SQuAD)",
                        "Method": f"ICL-{model}",
                        "Metric": "F1",
                        "Score": f1_score
                    })
        df = pd.DataFrame(data)
        st.dataframe(df)

        if "comparison_history" not in st.session_state:
            st.session_state["comparison_history"] = []
        st.session_state["comparison_history"].append(df)

        chart = alt.Chart(df).mark_bar().encode(
            x="Prompt Style:N",
            y="F1:Q",
            color="Model:N",
            tooltip=["Model", "Prompt Style", "F1"]
        ).facet(
            facet=alt.Facet("Model:N", title=None),
            columns=50
        ).properties(title="Prompt Style Impact on F1")

        st.altair_chart(chart, use_container_width=True)
