# prompts.py
import json

def build_prompt(context, question, style, task, n_shots=0):
    if task == "Question Answering (SQuAD)":
        if style == "Zero-Shot" or n_shots == 0:
            return f"""Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:"""

        elif style == "Few-Shot":
            with open("examples.json") as f:
                examples = json.load(f)[:n_shots]
            shots = ""
            for ex in examples:
                shots += f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
            return shots + f"Context: {context}\nQuestion: {question}\nAnswer:"

        elif style == "Chain-of-Thought":
            return f"""You are a helpful assistant. Think step-by-step before answering.

Context:
{context}

Question:
{question}

Let's think step-by-step:"""

    elif task == "Text Classification (AG News)":
        return f"""Classify the news headline into one of the following categories: World, Sports, Business, Sci/Tech.

Headline: {context}

Category:"""

    elif task == "Translation (WMT)":
        return f"""Translate the following sentence from English to French.

English: {context}
French:"""

    return context