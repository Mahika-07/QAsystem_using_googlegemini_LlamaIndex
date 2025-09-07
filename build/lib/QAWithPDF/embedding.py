# from llama_index.core import VectorStoreIndex
# from llama_index.core import StorageContext, load_index_from_storage
# from llama_index.embeddings.gemini import GeminiEmbedding
# from llama_index.core.settings import Settings
# from llama_index.core.node_parser import SentenceSplitter

# from QAWithPDF.data_ingestion import load_data
# from QAWithPDF.model_api import load_model

# import sys
# from exception import customexception
# from logger import logging

# def download_gemini_embedding(model, document):
#     """
#     Downloads and initializes a Gemini Embedding model for vector embeddings.

#     Returns:
#     - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
#     """
#     try:
#         logging.info("Initializing Gemini Embedding model...")
#         gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")

#         # Apply global settings instead of ServiceContext
#         Settings.llm = model
#         Settings.embed_model = gemini_embed_model
#         Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)

#         logging.info("Building vector store index...")
#         index = VectorStoreIndex.from_documents(document)
#         index.storage_context.persist()

#         logging.info("Creating query engine...")
#         query_engine = index.as_query_engine()
#         return query_engine

#     except Exception as e:
#         raise customexception(e, sys)


from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter

from QAWithPDF.data_ingestion import load_single_pdf
from QAWithPDF.model_api import load_model

import os
import sys
from exception import customexception
from logger import logging

INDEX_DIR = "indexes"

def get_or_create_index(pdf_filename, model):
    try:
        logging.info(f"📄 Starting index process for: {pdf_filename}")

        # Set paths
        index_path = os.path.join(INDEX_DIR, os.path.splitext(pdf_filename)[0])

        # ✅ Use Gemini embedding model explicitly
        embed_model = GeminiEmbedding(model_name="models/embedding-001")

        # Apply settings
        Settings.llm = model
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

        if os.path.exists(index_path):
            logging.info("📦 Loading index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
        else:
            logging.info("🆕 Creating new index...")
            document = load_single_pdf(f"uploaded_pdfs/{pdf_filename}")
            index = VectorStoreIndex.from_documents(document)
            index.storage_context.persist(persist_dir=index_path)

        logging.info("✅ Index ready.")

        # Instead of returning `index.as_query_engine()`, just return the index directly for further handling
        return index

    except Exception as e:
        raise customexception(e, sys)
