# In-Context Learning vs Fine-Tuning: Comparative Analysis on NLP Tasks

This project presents a comparative study between In-Context Learning (ICL) and Fine-Tuning on standard NLP tasks:
- Question Answering (SQuAD)
- Text Classification (AG News)
- Translation (WMT14 English to French)

The system evaluates multiple prompt styles and model types using metrics such as F1, Exact Match, ROUGE-L, Accuracy, Precision, Recall, BLEU, BERTScore, and ChrF. It is implemented using Streamlit, Hugging Face Transformers, and Ollama (for running models like LLaMA3 and DeepSeek-R1).

---

## Features

- Side-by-side evaluation of ICL and fine-tuned models
- Prompt-style comparison: Zero-shot, Few-shot, Chain-of-thought
- Few-shot scaling analysis for question answering tasks
- Visual summaries of Accuracy, Precision, Recall, F1, and other metrics
- Response time comparisons between models and prompt styles
- Live model output display for each task and setting
- LaTeX table and PNG graph export options

---

## Project Structure

```plaintext
icl_vs_finetuning/
├── app.py                    # Streamlit application
├── utils.py                  # Metric implementations
├── prompts.py                # Prompt template generator
├── examples.json             # Few-shot input examples
├── requirements.txt          # Python dependencies
├── icl_vs_ft_notebook.ipynb  # Supporting research notebook
├── results/                  # All screenshot result images


```

---

## Setup Instructions

### 🔧 1. Clone & Install
```bash
git clone https://github.com/your-repo/icl-vs-finetuning.git
cd icl-vs-finetuning
pip install -r requirements.txt
```

### 2. Install Ollama & Pull Models
```bash
ollama run llama3
ollama run deepseek
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## Dataset Sources

- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- [WMT14 EN-FR](http://www.statmt.org/wmt14/translation-task.html)

---

## Evaluation Metrics

| Task               | Metrics Used                             |
|--------------------|-------------------------------------------|
| QA (SQuAD)         | F1, Exact Match, ROUGE-L                  |
| Classification     | Accuracy, Precision, Recall, F1           |
| Translation        | BLEU, BERTScore, ChrF                    |

---

## Visual Outputs

All performance visualizations and results screenshots are located in the figures/ directory. This includes:

 - Metric bar charts (Accuracy, F1, Precision, Recall)

 - Prompt style impact comparisons

 - Few-shot scaling performance curves

 - ICL vs Fine-Tuned answer comparisons

 - Response time analysis

---

## Results Overview

The experimental results indicate that fine-tuned models consistently outperform in-context learners in terms of both accuracy and stability across tasks. However, ICL models demonstrated significant adaptability in few-shot and chain-of-thought scenarios, particularly in question answering. Ensemble fine-tuned models achieved the highest performance on AG News and SQuAD, while ICL models like LLaMA3 performed competitively with sufficient prompt tuning.

Key observations:
- Fine-tuned models achieved higher average F1 scores and lower variance
- Prompt style had a notable impact: Chain-of-thought prompts improved ICL QA performance
- Few-shot scaling showed diminishing returns beyond 4–6 shots
- Translation tasks yielded better BLEU and BERTScore under fine-tuning

These findings suggest that while fine-tuning yields the best raw performance, ICL is a viable and flexible alternative in constrained or rapid-deployment scenarios.

---