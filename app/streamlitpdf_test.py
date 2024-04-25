import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

from vllm_interface import llm

import os

# Function to format the documents
def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_local_file(uploaded_file):
    if uploaded_file:
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.get_value())
        return temp_file
                
def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # Load PDF documents using PyPDFLoader with text splitting
        pdf_loader = PyPDFLoader("./data/temp.pdf")
        pdf_documents = pdf_loader.load_and_split()

        # Select embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create a vectorstore from documents
        vector_store = Chroma.from_documents(pdf_documents, embeddings)
        retriever = vector_store.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query_text)
        
        # Initialize chat model
        chat_model = llm

        # Pull the RAG prompt
        from prompt_templates import template2
        prompt_template = template2
        output_parser = StrOutputParser()

        # Create a custom RAG chain
        rag_chain = RunnableParallel(
            {"context": lambda x: retrieved_docs, "question": RunnablePassthrough()}
        ) | prompt_template | chat_model | output_parser

        return rag_chain.stream(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload and create local file
uploaded_file = st.file_uploader('Upload an article', type='pdf')
if uploaded_file is not None:
  with open(os.path.join('./data', "temp.pdf"), 'wb') as output_temporary_file:
    output_temporary_file.write(uploaded_file.read())
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    #openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = st.write_stream(generate_response(output_temporary_file, query_text))
#             response = generate_response(uploaded_file, query_text)
#             result.append(response)

# if len(result):
#     st.info(response)