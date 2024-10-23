import boto3
import os
import json
from pypdf import PdfReader
import streamlit as st
import langchain
import langchain_community


## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.prompts import PromptTemplate


# 1st Method.
# Implement the AWS Textract for Extract the Text.

file_path = 'data\sample.pdf'
with open(file_path, 'rb') as file:
    bytes_data = file.read()

# AWS setup
session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name=''
)

text_client = session.client('textract')
response = text_client.analyze_document(
        Document={'Bytes': bytes_data},
        FeatureTypes=['FORMS', 'TABLES']
    
)

blocks = response['Blocks']
text = ''
for block in blocks:
    if block['BlockType'] == 'LINE':
        text += block['Text'] + '\n'




# Streamlit APP

def main():
    st.set_page_config("CHAT PDF")
    st.header("Chat With PDF using AWS Bedrock💁")

    uploded_file = st.file_uploader("Uplode a PDF file", type="pdf")

    # Save the Uploded Pdf and Extract the Context.
    if uploded_file is not None:
        st.write("Extracting the Text form the Pdf")
        save_dir = 'D:\personal\RAG project\data'
        save_path = os.path.join(save_dir, uploded_file.name)

        if os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploded_file.getbuffer())
                st.success("File saved")
        else:
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploded_file.getbuffer())
                st.success("Path Created File Saved")
        txt = data_ingestion()
        st.success("Text Extraction Compelted")

    user_question = st.text_input("Enter You Question")
    button = st.button("Enter")

    final_prompt = prompt(contex=data_ingestion(), question=user_question)

    answer = claude_llm(str(final_prompt))

    if button is True:
        if uploded_file is not None:
            st.write("Answer : ", answer)           
        else:
            st.write("Pless Uplode the File.")


# Prompt.

def prompt(contex, question):
    
    prompt_template = """

    Human: Use the following pieces of context to provide a 
    concise answer to the question at the end but usse atleast summarize with 
    50 words with detailed explaantions. If you don't know the answer, 
    just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    formated_prompt = PROMPT.format(context = contex, question = question)

    return formated_prompt



#Method 2.
# Extract the Text from the PDF

def data_ingestion():
    loader=PyPDFDirectoryLoader("D:\personal\RAG project\data")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs



# LLM (Anthorpic Claude Sonnet 3)

def claude_llm(prompt):
    print("1st step")
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    model = boto3.client(service_name='',
                region_name='',
                aws_access_key_id='',
                aws_secret_access_key='')
    
    body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 200,
    "temperature":0,
    "top_k":250,
    "top_p":1,
    "messages": [{
        "role": "user",
        "content": [{"type": "text", "text": str(prompt)}],
        }]
    })

    response = model.invoke_model_with_response_stream(body=body, modelId=model_id)

    generated_response = ""

    event_stream = response['body']

    for event in event_stream:
        chunk = event.get('chunk')

    if chunk.get("type") == "content_block_delta":
        if chunk["delta"].get("type") == "text_delta":
            generated_response += chunk["delta"]["text"]

    return generated_response



if __name__ == "__main__":
    main()