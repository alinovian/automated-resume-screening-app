import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load Pre-Trained Model, TF-IDF vectorizer, Label Encoder
svm_model = pickle.load(open('classifier_svm.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))


# Function to clean resume text
def cleanResume(text):
    cleanText = text.lower()
    cleanText = re.sub(r'http\S+|www\S+', '', cleanText)
    cleanText = re.sub(r'\b(rt|cc)\b', '', cleanText)
    cleanText = re.sub(r'#\S+', '', cleanText)
    cleanText = re.sub(r'@\S+', '', cleanText)
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7F]+', ' ', cleanText)
    cleanText = re.sub(r'\d+', '', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)

    vectorized_text = tfidf.transform([cleaned_text])

    vectorized_text = vectorized_text.toarray()

    predicted_category = svm_model.predict(vectorized_text)

    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]


# Streamlit app layout
def main():
    st.set_page_config(page_title="Automated Resume Screening App", page_icon="🚀", layout="wide")

    # Header Section
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Automated Resume Category Prediction</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<p style='text-align: center;'>Upload your resume in PDF, DOCX, or TXT format to predict the job category.</p>", unsafe_allow_html=True)

    # Layout: File Upload and Display
    with st.container():
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📂 Upload Resume")
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

        with col2:
            if uploaded_file is not None:
                st.markdown("### 📜 Extracted Resume Text")
                try:
                    resume_text = handle_file_upload(uploaded_file)
                    st.success("File uploaded and processed successfully!")
                    st.text_area("Extracted Text", resume_text, height=200)

                except Exception as e:
                    st.error(f"Error processing the file: {str(e)}")

    # Prediction Section
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Predicted Category</h3>", unsafe_allow_html=True)
        category = pred(resume_text)
        st.markdown(
            f"<h2 style='text-align: center; color: #FF5733;'>🎯 {category}</h2>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
