# Use to upload a single file on Streamlit and get a single answer for the selected query
# Streamlit UI

# Import necessary libraries
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from utils import create_vector_db, load_model_and_tokenizer, streamlit_parse_file, question_answering, save_uploaded_file
import configs
import torch

# Cached function to load model and tokenizer
@st.cache_resource()
def cached_load_model_and_tokenizer(model_name_or_path):
    return load_model_and_tokenizer(model_name_or_path)

# Streamlit app title
st.title("Metadata Generation")

# File upload section
uploaded_file = st.file_uploader("Upload a file", type=["pptx", "docx", "xlsx", "xls", "csv", "ipynb", "py", "md", "pdf"])

if uploaded_file is not None:
    st.success("File successfully uploaded!")
    try:
        texts = streamlit_parse_file(uploaded_file)
        if texts:
            st.write("Files parsed successfully")
            save_uploaded_file(uploaded_file, configs.UPLOADED_FILE_PATH)
            st.write("Files uploaded successfully")
        else:
            st.error("No file parsed")
            st.stop()
        
        st.write("\nCreating vector db...")
        try:
            create_vector_db(texts)
        except Exception as e:
            st.error(f"Failed to create vector db: {e}")
            st.stop()

        st.write("Vector db created successfully.")        
        
        # Load model, tokenizer, and vector database
        st.write("\nLoading model, tokenizer, and db...")
        loaded_pipeline =cached_load_model_and_tokenizer(configs.model_name_or_path)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cuda'})
        db = FAISS.load_local(configs.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        st.write("Model, tokenizer, and db loaded successfully.")

        # Define the question to be answered
        questions = "Short description of the file"

        # Answer the selected question using LLM
        try:
            st.write("\n Question:\n", questions)

            llm_answer = question_answering(user_query=questions, pipeline=loaded_pipeline, db=db)

            st.write("Answer:")
            st.write(llm_answer)
            torch.cuda.empty_cache()
        except Exception as e:
            st.error(f"An unexpected error occurred during question answering: {e}")

    except Exception as e:
        st.error(f"An unexpected error occurred during execution: {e}")   

