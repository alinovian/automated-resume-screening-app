import numpy as np
import streamlit as st
import pickle
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

nltk.download('punkt')
nltk.download('stopwords')

# Load Models
with open("useless/word2vec_model.pkl", "rb") as file:
    word2vec_model = pickle.load(file)
with open("useless/clf.pkl", "rb") as file:
    clf = pickle.load(file)
with open("useless/encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# import joblib
# clf = joblib.load("clf.pkl")

# Text cleaning function
def clean_text(resume_text):
    cleanText = resume_text.lower()
    cleanText = re.sub(r'http\S+|www\S+', '', cleanText)
    cleanText = re.sub(r'\b(rt|cc)\b', '', cleanText)
    cleanText = re.sub(r'#\S+', '', cleanText)
    cleanText = re.sub(r'@\S+', '', cleanText)
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7F]+', ' ', cleanText)
    cleanText = re.sub(r'\d+', '', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText

# Tokenization and stopword removal
def tokenize_text(text):
    return word_tokenize(text)

sw = stopwords.words('indonesian')

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in sw]

def get_doc_vector(tokens, word2vec_model):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

# Read text from file
def read_file(upload_file):
    try:
        if upload_file.type == "application/pdf":
            reader = PdfReader(upload_file)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif upload_file.type == "text/plain":
            text = upload_file.read().decode('utf-8', errors='ignore')
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")
        if not text.strip():
            raise ValueError("The uploaded file does not contain readable text.")
        return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Web app main function
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if upload_file is not None:
        resume_text = read_file(upload_file)
        if not resume_text:
            return  # Stop execution if there's an error in reading the file

        try:
            cleaned_text = clean_text(resume_text)
            tokenized_text = tokenize_text(cleaned_text)
            removed_stopwords = remove_stopwords(tokenized_text)

            embeddings = []
            for word in removed_stopwords:
                if word in word2vec_model.wv:
                    embeddings.append(word2vec_model.wv[word])

            if not embeddings:
                st.error("No valid embeddings found for the input text. Please try with a different resume.")
                return

            avg_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
            predicted_id = clf.predict(avg_embedding)
            predicted_label = label_encoder.inverse_transform(predicted_id)[0]

            st.success(f"The predicted category for this resume is: {predicted_label}")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# Run the app
if __name__ == "__main__":
    main()