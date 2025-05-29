# 🧠 In-Context Learning vs Fine-Tuning: Comparative Analysis on NLP Tasks

This project compares **In-Context Learning (ICL)** and **Fine-Tuning** on standard NLP tasks: 
- **Question Answering** (SQuAD)
- **Text Classification** (AG News)
- **Translation** (WMT - English to French)

We evaluate performance using multiple metrics: **F1, Exact Match, ROUGE-L, Accuracy, Precision, Recall, BLEU, BERTScore,** and **ChrF**. The app is built with **Streamlit**, integrated with **Hugging Face Transformers**, and **Ollama** (for LLaMA3 and DeepSeek-R1).

---

## 🚀 Features

- Side-by-side comparison of ICL and fine-tuned models
- Visualize metrics with Altair charts
- Prompt-style experiments: Zero-shot, Few-shot, Chain-of-thought
- Few-shot scaling effect for QA
- Upload your own CSV for batch evaluation
- Export results as CSV/PNG
- LaTeX table generator for reports

---

## 📁 Project Structure

```plaintext
icl_vs_finetuning/
├── app.py                    # Streamlit frontend
├── utils.py                  # All evaluation metric functions
├── prompts.py               # Prompt generation logic
├── examples.json            # Few-shot examples for QA
├── requirements.txt         # Python dependencies
```

---

## 📦 Setup Instructions

### 🔧 1. Clone & Install
```bash
git clone https://github.com/your-repo/icl-vs-finetuning.git
cd icl-vs-finetuning
pip install -r requirements.txt
```

### 🤖 2. Install Ollama & Pull Models
```bash
ollama run llama3
ollama run deepseek
```

### ▶️ 3. Run the App
```bash
streamlit run app.py
```

---

## 📊 Dataset Sources

- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- [WMT14 EN-FR](http://www.statmt.org/wmt14/translation-task.html)

---

## 🧪 Evaluation Metrics

| Task               | Metrics Used                             |
|--------------------|-------------------------------------------|
| QA (SQuAD)         | F1, Exact Match, ROUGE-L                  |
| Classification     | Accuracy, Precision, Recall, F1           |
| Translation        | BLEU, BERTScore, ChrF                    |

---

## 📂 Custom Dataset Format

Upload a CSV file with the following headers:
```csv
task,context,question,expected,prompt_style
Question Answering (SQuAD),"Python is a language.","What is Python?","a language","Few-Shot"
```

---

## 📄 Citation

If you use this project, please cite:

```
@misc{icl-vs-finetuning-2025,
  title={In-Context Learning vs Fine-Tuning: Comparative Analysis on NLP Tasks},
  author={Your Team},
  year={2025},
  note={CSC 7825 - Machine Learning Course Project}
}
```

---

## 💬 Contact

For questions, reach out via GitHub Issues or contact [youremail@example.com].
