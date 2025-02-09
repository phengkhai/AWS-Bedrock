import json
import os
import sys
import boto3
import streamlit as st
import logging
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion
def data_ingestion():
    try:
        loader = PyPDFDirectoryLoader("data")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        st.error("Failed to load documents. Please check the data directory and try again.")
        return None

# Vector Embedding and vector store
def get_vector_store(docs):
    try:
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local("faiss_index")
        logger.info("Vector store created and saved locally.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error("Failed to create vector store. Please try again.")

# Initialize Claude LLM
def get_claude_llm():
    try:
        llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
        return llm
    except Exception as e:
        logger.error(f"Error initializing Claude LLM: {e}")
        st.error("Failed to initialize Claude LLM. Please check your AWS credentials and try again.")
        return None

# Initialize Llama2 LLM
def get_llama2_llm():
    try:
        llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
        return llm
    except Exception as e:
        logger.error(f"Error initializing Llama2 LLM: {e}")
        st.error("Failed to initialize Llama2 LLM. Please check your AWS credentials and try again.")
        return None

# Prompt template for the LLM
prompt_template = """
Human: You are an advanced AI assistant tasked with answering questions based on the provided context. 

Please follow these guidelines:  
- Provide a **detailed and well-structured answer** with at least **250 words**.  
- Use **clear explanations, relevant examples, and logical reasoning** to support your response.  
- If the information is unavailable in the given context, **clearly state that you don’t know** rather than making up an answer.  

<context>
{context}
</context>

Now, based on the above context, **answer the following question**:

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Get response from LLM
def get_response_llm(llm, vectorstore_faiss, query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer = qa({"query": query})
        return answer['result']
    except Exception as e:
        logger.error(f"Error getting response from LLM: {e}")
        st.error("Failed to get response from LLM. Please try again.")
        return None

# Main function
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                if docs:
                    get_vector_store(docs)
                    st.success("Vector store updated successfully!")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            try:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
                llm = get_claude_llm()
                if llm:
                    response = get_response_llm(llm, faiss_index, user_question)
                    if response:
                        st.write(response)
                        st.success("Done")
            except Exception as e:
                logger.error(f"Error during Claude Output: {e}")
                st.error("Failed to generate Claude Output. Please try again.")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            try:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
                llm = get_llama2_llm()
                if llm:
                    response = get_response_llm(llm, faiss_index, user_question)
                    if response:
                        st.write(response)
                        st.success("Done")
            except Exception as e:
                logger.error(f"Error during Llama2 Output: {e}")
                st.error("Failed to generate Llama2 Output. Please try again.")

if __name__ == "__main__":
    main()