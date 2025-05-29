from sklearn.metrics import precision_score, recall_score, f1_score
from evaluate import load

# Load evaluation modules
squad_metric = load("squad")
bleu_metric = load("bleu")
rouge_metric = load("rouge")
bertscore_metric = load("bertscore")
chrf_metric = load("chrf")

# ============================
# Question Answering Metrics
# ============================
def compute_f1(pred, true):
    return squad_metric.compute(
        predictions=[{"prediction_text": pred, "id": "0"}],
        references=[{"answers": {"text": [true], "answer_start": [0]}, "id": "0"}]
    )["f1"]

def compute_em(pred, true):
    return squad_metric.compute(
        predictions=[{"prediction_text": pred, "id": "0"}],
        references=[{"answers": {"text": [true], "answer_start": [0]}, "id": "0"}]
    )["exact_match"]

def compute_rouge(pred, true):
    return rouge_metric.compute(predictions=[pred], references=[true])["rougeL"] * 100

# ============================
# Classification Metrics
# ============================
def compute_accuracy(pred, true):
    return int(pred == true) * 100

def compute_classification_metrics(pred, true):
    pred = pred.strip().lower()
    true = true.strip().lower()

    # Extract known class label from prediction text
    labels = ["world", "sports", "business", "sci/tech"]
    matched = [label for label in labels if label in pred]
    if matched:
        pred = matched[0]
    else:
        pred = "unknown"

    if true not in labels:
        true = "unknown"

    return {
        "Accuracy": int(pred == true) * 100,
        "Precision": precision_score([true], [pred], labels=labels + ["unknown"], average='macro', zero_division=0) * 100,
        "Recall": recall_score([true], [pred], labels=labels + ["unknown"], average='macro', zero_division=0) * 100,
        "F1": f1_score([true], [pred], labels=labels + ["unknown"], average='macro', zero_division=0) * 100
    }

# ============================
# Translation Metrics
# ============================
def compute_bleu(pred, true):
    return bleu_metric.compute(predictions=[pred], references=[true])["bleu"] * 100

def compute_bertscore(pred, true):
    result = bertscore_metric.compute(predictions=[pred], references=[true], lang="fr")
    return result["f1"][0] * 100

def compute_chrf(pred, true):
    return chrf_metric.compute(predictions=[pred], references=[true])["score"]

# ============================
# Utility for LaTeX Table
# ============================
def generate_latex_table(df):
    if df.empty:
        return "No data to generate LaTeX."
    cols = df.columns.tolist()
    latex = "\\begin{tabular}{|" + " | ".join(["c"] * len(cols)) + "|}\n\\hline\n"
    latex += " & ".join(cols) + " \\\\\n\\hline\n"
    for _, row in df.iterrows():
        row_data = " & ".join(str(row[col]) for col in cols)
        latex += f"{row_data} \\\\\n\\hline\n"
    latex += "\\end{tabular}"
    return latex
