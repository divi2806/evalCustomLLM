import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from detoxify import Detoxify
import gspread
from google.oauth2.service_account import Credentials
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# File reading functions
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_text(file):
    return file.getvalue().decode("utf-8")

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_google_sheet(sheet_url):
    creds = Credentials.from_service_account_file("path/to/your/credentials.json")
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url).sheet1
    data = sheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df.to_string()

# Streamlit GUI
st.set_page_config(page_title="Zelo QA Evaluation")
st.header("Zelo QA Evaluation")

file_type = st.selectbox("Select file type", ["PDF", "Excel", "Word", "Plain Text", "CSV", "Google Sheets"])

if file_type == "Google Sheets":
    sheet_url = st.text_input("Enter Google Sheets URL")
    if sheet_url:
        text = read_google_sheet(sheet_url)
else:
    uploaded_file = st.file_uploader(f"Upload your {file_type} file", type=["pdf", "xlsx", "docx", "txt", "csv"])
    
    if uploaded_file is not None:
        if file_type == "PDF":
            text = read_pdf(uploaded_file)
        elif file_type == "Excel":
            text = read_excel(uploaded_file)
        elif file_type == "Word":
            text = read_docx(uploaded_file)
        elif file_type == "Plain Text":
            text = read_text(uploaded_file)
        elif file_type == "CSV":
            text = read_csv(uploaded_file)

if 'text' in locals():
    # Process the text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    # New section for uploading questions, ground truths, and Zelo answers
    st.subheader("Upload Evaluation Data")
    questions_file = st.file_uploader("Upload questions file", type=["txt"])
    ground_truths_file = st.file_uploader("Upload ground truths file", type=["txt"])
    zelo_answers_file = st.file_uploader("Upload Zelo answers file", type=["txt"])
    
    if questions_file and ground_truths_file and zelo_answers_file:
        questions = questions_file.getvalue().decode("utf-8").split("\n")
        ground_truths = ground_truths_file.getvalue().decode("utf-8").split("\n")
        zelo_answers = zelo_answers_file.getvalue().decode("utf-8").split("\n")
        
        # Remove any empty lines and strip whitespace
        questions = [q.strip() for q in questions if q.strip()]
        ground_truths = [gt.strip() for gt in ground_truths if gt.strip()]
        zelo_answers = [za.strip() for za in zelo_answers if za.strip()]
        
        # Ensure all files have the same number of entries
        if len(questions) == len(ground_truths) == len(zelo_answers):
            st.write(f"Loaded {len(questions)} sets of questions, ground truths, and Zelo answers.")
        else:
            st.error("Error: The number of entries in the uploaded files do not match. Please ensure all files have the same number of lines.")
            st.stop()

        def calculate_metrics(zelo_answer, ground_truth, question):
            # Similarity (Cosine Similarity with Ground Truth)
            similarity_score = cosine_similarity([embeddings.embed_query(zelo_answer)], [embeddings.embed_query(ground_truth)])[0][0]
            
            # Exact Match (SequenceMatcher)
            exact_match_score = SequenceMatcher(None, zelo_answer, ground_truth).ratio()
            
            # Faithfulness (using the improved calculation)
            if similarity_score > 0.9:
                faithfulness_score = 1.0
            elif similarity_score > 0.5:
                faithfulness_score = 0.9 + (similarity_score - 0.5) * (1 - 0.9) / 0.4
            else:
                faithfulness_score = 0.0
            
            # Relevance (Cosine Similarity with Question)
            relevance_score = cosine_similarity([embeddings.embed_query(zelo_answer)], [embeddings.embed_query(question)])[0][0]
            
            # Hallucination (Derived from Faithfulness Score)
            hallucination_score = 1 - faithfulness_score
            
            # QA Accuracy (same as Exact Match for this implementation)
            qa_accuracy_score = exact_match_score
            
            # Toxicity (Detoxify)
            toxicity = Detoxify('original').predict(zelo_answer)
            toxicity_score = toxicity['toxicity']
            
            return {
                "similarity": similarity_score,
                "exact_match": exact_match_score,
                "faithfulness": faithfulness_score,
                "relevance": relevance_score,
                "hallucination": hallucination_score,
                "qa_accuracy": qa_accuracy_score,
                "toxicity": toxicity_score
            }

        # Calculate metrics for each question and display
        all_metrics = []
        for q, gt, za in zip(questions, ground_truths, zelo_answers):
            metrics = calculate_metrics(za, gt, q)
            all_metrics.append(metrics)
            
            st.write(f"\nQuestion: {q}")
            st.write(f"Ground Truth: {gt}")
            st.write(f"Zelo Answer: {za}")
            st.write("Metrics:")
            for metric, value in metrics.items():
                st.write(f"{metric.capitalize()}: {value:.4f}")

        # Calculate and display aggregate metrics
        aggregate_metrics = {metric: np.mean([m[metric] for m in all_metrics]) for metric in all_metrics[0].keys()}

        st.write("\nAggregate Metrics:")
        st.write(aggregate_metrics)

        # Plotting
        def plot_metrics(all_metrics):
            data = pd.DataFrame(all_metrics)
            
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            
            metrics = list(all_metrics[0].keys())
            for i, metric in enumerate(metrics):
                row = i // 3
                col = i % 3
                sns.histplot(data[metric], ax=axes[row, col], kde=True)
                axes[row, col].set_title(f"{metric.capitalize()} Scores")
            
            if len(metrics) < 9:
                for i in range(len(metrics), 9):
                    row = i // 3
                    col = i % 3
                    fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            return fig

        fig = plot_metrics(all_metrics)
        st.pyplot(fig)

        # Save results to CSV
        df = pd.DataFrame(all_metrics)
        df['question'] = questions
        df['ground_truth'] = ground_truths
        df['zelo_answer'] = zelo_answers
        results_csv = 'zelo_evaluation_results.csv'
        df.to_csv(results_csv, index=False)
        st.write(f"\nDetailed results saved to '{results_csv}'")

else:
    st.write("Please upload a file or enter a valid Google Sheets URL.")
