# import streamlit as st
# from QAWithPDF.data_ingestion import load_data
# from QAWithPDF.embedding import download_gemini_embedding
# from QAWithPDF.model_api import load_model

    
# def main():
#     st.set_page_config("QA with Documents")
    
#     doc=st.file_uploader("upload your document")
    
#     st.header("QA with Documents(Information Retrieval)")
    
#     user_question= st.text_input("Ask your question")
    
#     if st.button("submit & process"):
#         with st.spinner("Processing..."):
#             document=load_data(doc)
#             model=load_model()
#             query_engine=download_gemini_embedding(model,document)
                
#             response = query_engine.query(user_question)
                
#             st.write(response.response)
                
                
# if __name__=="__main__":
#     main()          
                
    
    
    
    
import streamlit as st
import os

from QAWithPDF.embedding import get_or_create_index
from QAWithPDF.model_api import load_model

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


    if st.button("Submit & Process") and selected_pdf and user_question:
        with st.spinner("Retrieving answer..."):
            model = load_model()
            index = get_or_create_index(selected_pdf, model)

            # Create query engine after getting the index
            query_engine = index.as_query_engine(similarity_top_k=5, mode="embedding")

            response = query_engine.query(user_question)
            st.markdown("### 💬 Answer:")
            st.write(response.response)

if __name__ == "__main__":
    main()
