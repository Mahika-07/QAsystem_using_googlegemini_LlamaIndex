import pandas as pd
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
import numpy as np
import torch 
torch.classes.__path__ = []

def load_ground_truth(csv_path="test_data.csv"):
    return pd.read_csv(csv_path)

def normalize_question(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # collapse multiple spaces
    return text


def find_ground_truth(df, pdf_filename, question):
    df.columns = df.columns.str.strip()
    df['Files'] = df['Files'].astype(str).str.strip().str.lower()
    df['Question'] = df['Question'].astype(str).apply(normalize_question)

    file_id = pdf_filename.replace(".pdf", "").strip().lower()
    normalized_q = normalize_question(question)

    row = df[(df['Files'] == file_id) & (df['Question'] == normalized_q)]

    if not row.empty:
        return row.iloc[0]['Ground Truth']
    return None

    
def bertscore_accuracy(pred, gt):
    _, _, F1 = bert_score([pred], [gt], lang="en", verbose=False)
    return int(F1[0].item() > 0.85)

def semantic_accuracy(pred, gt):
    return int(fuzz.token_sort_ratio(pred, gt) > 85)

def compute_jaccard_similarity(ans1, ans2):
    set1 = set(ans1.lower().split())
    set2 = set(ans2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def compute_metrics(predicted, ground_truth):
    metrics = {}
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    try:
        rouge_scores = scorer.score(ground_truth, predicted)
    except ValueError:
        rouge_scores = {"rouge1": None, "rouge2": None, "rougeL": None}

    # ROUGE F1, Precision, Recall
    for key in ['rouge1', 'rouge2', 'rougeL']:
        if rouge_scores[key]:
            metrics[f"{key.upper()}_F1"] = round(rouge_scores[key].fmeasure, 4)
            metrics[f"{key.upper()}_Precision"] = round(rouge_scores[key].precision, 4)
            metrics[f"{key.upper()}_Recall"] = round(rouge_scores[key].recall, 4)
        else:
            metrics[f"{key.upper()}_F1"] = 0
            metrics[f"{key.upper()}_Precision"] = 0
            metrics[f"{key.upper()}_Recall"] = 0

    # BLEU
    smoothie = SmoothingFunction().method4
    metrics['BLEU'] = sentence_bleu([ground_truth.split()], predicted.split(), smoothing_function=smoothie)

    # Jaccard
    metrics['Jaccard'] = compute_jaccard_similarity(predicted, ground_truth)

    # BERTScore (F1, Precision, Recall)
    try:
        P, R, F1 = bert_score([predicted], [ground_truth], lang="en", verbose=False)
        metrics['BERTScore_F1'] = F1[0].item()
        metrics['BERTScore_Precision'] = P[0].item()
        metrics['BERTScore_Recall'] = R[0].item()
    except Exception:
        metrics['BERTScore_F1'] = 0.0
        metrics['BERTScore_Precision'] = 0.0
        metrics['BERTScore_Recall'] = 0.0

    # BERTScore-based Accuracy
    metrics['Accuracy'] = bertscore_accuracy(predicted, ground_truth)

    return metrics
