# In-Context Learning vs Fine-Tuning: Comparative Analysis on NLP Tasks

This project presents a comparative study between In-Context Learning (ICL) and Fine-Tuning on standard NLP tasks:
- Question Answering (SQuAD)
- Text Classification (AG News)
- Translation (WMT14 English to French)

The system evaluates multiple prompt styles and model types using metrics such as F1, Exact Match, ROUGE-L, Accuracy, Precision, Recall, BLEU, BERTScore, and ChrF. It is implemented using Streamlit, Hugging Face Transformers, and Ollama (for running models like LLaMA3 and DeepSeek-R1).

---

## Features

- Side-by-side evaluation of ICL and fine-tuned models
- Comparison of Zero-shot, Few-shot, and Chain-of-thought prompting styles
- Few-shot scaling analysis for QA tasks
- Custom dataset support via CSV upload
- Real-time model output and performance visualization
- LaTeX table generation for report inclusion

---

## Project Structure

```plaintext
icl_vs_finetuning/
├── app.py                    # Streamlit frontend
├── utils.py                  # Evaluation metric implementations
├── prompts.py                # Prompt generation logic
├── examples.json             # Few-shot examples for QA
├── requirements.txt          # Python dependencies

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


