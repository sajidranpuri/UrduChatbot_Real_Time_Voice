import streamlit as st
import os
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Set Google Gemini API key explicitly
genai.configure(api_key="GOOGLE_GEMINI_API_KEY")  # Replace with your actual API key

# Streamlit app title
st.title("Urdu Intelligent Chatbot (Real-Time Voice)")
st.write("Click the button given below and speak near to microphone in Urdu!")

# Load PDF files and create a knowledge base
def load_knowledge_base(pdf_path):
    loader = PyPDFLoader('data.pdf')      # Replace with your PDF file path with forward slash
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_documents(texts, embeddings)
    return knowledge_base

# Initialize the knowledge base
pdf_path = r"data.pdf"                    # Replace with your PDF file path
knowledge_base = load_knowledge_base(pdf_path)

# Function to convert text to Urdu audio
def text_to_audio(text, output_file="output.mp3"):
    tts = gTTS(text, lang="ur")
    tts.save(output_file)
    return output_file

# Function to capture microphone audio and convert to text
def audio_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        status_placeholder = st.empty()
        status_placeholder.write("Speak now...")
        audio = r.listen(source)
        status_placeholder.empty()  # Clear message

        try:
            text = r.recognize_google(audio, language="ur-PK")
            return text
        except sr.UnknownValueError:
            return "Audio not understood"
        except sr.RequestError:
            return "API unavailable"

# Function to retrieve relevant text from the PDF
def retrieve_from_pdf(query):
    # Retrieve the most relevant chunks from the PDF
    docs = knowledge_base.similarity_search(query, k=2)  # Retrieve top 2 relevant chunks
    return docs

# Function to generate a response using Google Gemini
def generate_response(query, context):
    
    if not context.strip():  # Check if context is empty or just whitespace
        return "I couldn't find relevant information in the document. Please try a different question."
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    # Combine the query and context for the model
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text


# Streamlit UI components
if st.button("Start Recording"):
    # Capture audio and convert to text
    user_input = audio_to_text()
    
    if user_input and user_input not in ["Audio not understood", "API unavailable"]:
        st.write(f"User Input: {user_input}")

        # Step 1: Retrieve relevant text from the PDF
        retrieved_docs = retrieve_from_pdf(user_input)
        context = "\n".join([doc.page_content for doc in retrieved_docs])  # Combine retrieved chunks

        # Display the retrieved context (for verification)
        st.write("Retrieved Context from PDF:")
        st.write(context)

        # Convert retrieved context to Urdu audio
        context_audio_file = text_to_audio(context, "context_output.mp3")
        st.write("Retrieved Context (Audio):")
        st.audio(context_audio_file, format="audio/mp3")

        # Step 2: Generate response using the retrieved context
        response_text = generate_response(user_input, context)
        st.write(f"Chatbot Response (Text): {response_text}")

        # Convert chatbot response to Urdu audio
        response_audio_file = text_to_audio(response_text, "response_output.mp3")
        st.write("Chatbot Response (Audio):")
        st.audio(response_audio_file, format="audio/mp3")

    elif user_input in ["Audio not understood", "API unavailable"]:
        st.error(f"Error: {user_input}")