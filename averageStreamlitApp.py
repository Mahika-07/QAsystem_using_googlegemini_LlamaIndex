import streamlit as st
import os
import pandas as pd
import json
import nltk
import numpy as np  # Add this import to fix the error
from Evaluation import load_ground_truth, find_ground_truth, compute_metrics
from QAWithPDF.embedding import get_or_create_index
from QAWithPDF.model_api import load_model

# Download NLTK tokenizer if not already present
nltk.download('punkt', quiet=True)

UPLOAD_DIR = "uploaded_pdfs"

def save_uploaded_file(uploaded_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name

def get_all_uploaded_pdfs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]

def main():
    st.set_page_config("QA with Documents")
    st.title("📄 Multi-PDF QA System (Gemini + LlamaIndex)")

    # Upload new PDF
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        saved_name = save_uploaded_file(uploaded_file)
        st.success(f"{saved_name} uploaded and indexed!")

    # Show available PDFs
    pdf_list = get_all_uploaded_pdfs()
    selected_pdf = st.selectbox("Select a PDF to query:", pdf_list)

    user_question = st.text_input("Ask your question about the selected PDF:")

    overall_metrics = {
        "ROUGE-1": [],
        "ROUGE-2": [],
        "ROUGE-L": [],
        "BLEU": [],
        "Jaccard": [],
        "BERTScore": [],
        "Accuracy": []
    }

    if st.button("Submit & Process") and selected_pdf and user_question:
        with st.spinner("Retrieving answer..."):
            model = load_model()
            index = get_or_create_index(selected_pdf, model)

            # Create query engine after getting the index
            query_engine = index.as_query_engine(similarity_top_k=5, mode="embedding")

            response = query_engine.query(user_question)
            st.markdown("### 💬 Answer:")
            st.write(response.response)

            # Load ground truth CSV
            gt_df = load_ground_truth()

            ground_truth = find_ground_truth(gt_df, selected_pdf, user_question)
            if ground_truth:
                metrics = compute_metrics(response.response, ground_truth)
                st.markdown("### 📊 Evaluation Metrics for this Question:")
                st.json(metrics)

                # Collect metrics for overall evaluation
                for key in overall_metrics:
                    overall_metrics[key].append(metrics.get(key, 0))

            else:
                st.warning("⚠️ No matching ground truth found for this question in the CSV.")

        # Compute and display overall evaluation summary
        # if overall_metrics["ROUGE-1"]:
        #     st.markdown("### 📊 Overall Model Evaluation Summary:")

        #     # Calculate averages for each metric
        #     avg_metrics = {key: round(np.mean(value), 4) for key, value in overall_metrics.items()}

        #     st.json(avg_metrics)
        #     # Optionally, display the summary as a table
        #     st.write("#### Average Metrics Across All Questions:")
        #     avg_table = pd.DataFrame([avg_metrics])
        #     st.dataframe(avg_table)

if __name__ == "__main__":
    main()
