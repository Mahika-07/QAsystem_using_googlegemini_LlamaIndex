# from llama_index.core import SimpleDirectoryReader
# import sys
# from exception import customexception
# from logger import logging

# def load_data(data):
#     """
#     Load PDF documents from a specified directory.

#     Parameters:
#     - data (str): The path to the directory containing PDF files.

#     Returns:
#     - A list of loaded PDF documents. The specific type of documents may vary.
#     """
#     try:
#         logging.info("data loading started...")
#         loader = SimpleDirectoryReader("Data")
#         documents=loader.load_data()
#         logging.info("data loading completed...")
#         return documents
#     except Exception as e:
#         logging.info("exception in loading data...")
#         raise customexception(e,sys)



from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception
from logger import logging

def load_single_pdf(file_path):
    """
    Load a single PDF document from a given path.
    """
    try:
        logging.info(f"Loading PDF from {file_path}")
        loader = SimpleDirectoryReader(input_files=[file_path])
        return loader.load_data()
    except Exception as e:
        raise customexception(e, sys)
