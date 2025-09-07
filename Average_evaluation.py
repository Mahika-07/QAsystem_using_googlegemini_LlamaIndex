import pandas as pd
from QAWithPDF.embedding import get_or_create_index
from Evaluation import compute_metrics, load_ground_truth, find_ground_truth
from QAWithPDF.model_api import load_model

from tqdm import tqdm
from collections import defaultdict
import numpy as np

def evaluate_model_across_all_questions():
    # Load CSV with test questions and ground truths
    df = load_ground_truth("test_data.csv")
    df.columns = df.columns.str.strip().str.lower()  # normalize column names to lowercase and remove spaces

    # Rename for consistency
    df.rename(columns={
        'files': 'Files',
        'question': 'Question',
        'ground truth': 'Ground Truth'
    }, inplace=True)

    # Load your LLM model
    model = load_model()  # assumes Gemini or similar is returned
    print("🔍 Evaluating model...\n")

    # Collect all scores
    aggregated_metrics = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="📄 Processing"):
        pdf_filename = row['Files'].strip() + ".pdf"
        question = row['Question']
        ground_truth = row['Ground Truth']

        # Skip if ground truth is missing
        if pd.isna(ground_truth) or str(ground_truth).strip() == "":
            print(f"⚠️ Skipping: No ground truth for {pdf_filename} - Q: {question}")
            continue

        try:
            # Build or load index
            index = get_or_create_index(pdf_filename, model)
            query_engine = index.as_query_engine()

            # Get model prediction
            response = query_engine.query(question)
            predicted_answer = str(response).strip()

            # Compute metrics
            metrics = compute_metrics(predicted_answer, ground_truth)


            # Add metrics to aggregation
            for metric_name, value in metrics.items():
                aggregated_metrics[metric_name].append(value)

        except Exception as e:
            print(f"⚠️ Error with file {pdf_filename} & question: '{question}': {e}")
            continue

    # Compute average metrics
    print("\n📊 Average Evaluation Metrics Across All Questions:")
    for metric_name, scores in aggregated_metrics.items():
        avg = np.mean(scores)
        print(f"{metric_name}: {round(avg, 4)}")

if __name__ == "__main__":
    evaluate_model_across_all_questions()
