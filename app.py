import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv() 

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_file): #With this function I basically telling python to read all the files and its data that is present within the PDF files uploaded, extract the text and store it.
    text=''
    for pdf in pdf_file:
        pdf_reader = PdfReader(pdf)
        #In a PDF multiple pages may exist and what we want to do is when these pages are read it should be able to get the details and store it in a list. So, we will create another loop
        for page in pdf_reader.pages: #here the .pages argument implies going through each of the page in the pdf
            text += page.extract_text()
    
    return text

def get_text_chunks(text): #This function allows us to create chunks of our extracted text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) #we are going to use the recursive text splitter that we imported from langchain that is going to divide the text into chunks pf 10000 words with 1000 overlap and store it
    chunks = text_splitter.split_text(text) #Here we are taking the text present in the function argument and splitting it into chunks and storing it in a list based on the chunks we have described.
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) #This command we are basically telling FAISS to take in all the text chunks and embed based on the model that I have initialized
    vector_store.save_local('faiss_index') #For now I am saving in a local environment, but we can also save this to a database
    
    
def get_conversational_chain():
    #below we are providing the AI with context of what it needs to do and how it needs to provide answers. We are also telling it what to say incase it does not know the answer
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available for the context that has been provided", don't provide the wrong answer\n\n
    
    Context:\n {context}?\n
    Question: {question}\n
    
    Answer:
    """
    model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)  #Describing the model that we would be using
    prompt = PromptTemplate(template=prompt_template,input_variables=["Context","Question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt) #Chain_type basically allows us for document summarization
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index',embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {'inputdocuments':docs,'question':user_question},
        return_only_outputs=True
    )
    
    print (response)
    st.write('Reply:', response['output_text'])
    
#Creating our streamlit app
def main():
    st.set_page_config('Chat with PDF with Gemini')
    st.header("Chat with PDF using Gemini 🚀")
    
    user_question = st.text_input("Ask a question from the PDF file(s)")
    
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title('Menu:')
        pdf_docs = st.file_uploader("Upload PDF files and click on Submit", accept_multiple_files=True)
        
        if st.button("Submit"):
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success('Done')
                
                
if __name__ == '__main__':
    main()