import streamlit as st
import PyPDF2
import re
import tempfile
from nltk.tokenize import sent_tokenize
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import codecs
import os
import time
import pinecone_embeddings
from pinecone_embeddings import *
metadata_filepath = 'Digital X Solution Catalog Metadata for Hack to the Rescue.xlsx'
os.OPENAI_API_KEY ="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"
openai_api_key = "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"
def extract_data_from_headlines(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        data = []
        current_headline = None
        current_text = ""
        print("len", len(pdf_reader.pages))
        for page in pdf_reader.pages:
            text = page.extract_text()
            #print(text)
            data.append(text)
        return data


def fragment_text(text):
    sentences = sent_tokenize(text)
    return sentences
def fragment_text_for_summary(text):
    print("hello")
def fix_text_problems(text):
    #print("text",text)
    text = codecs.encode(text,"UTF-16")
    #text = codecs.decode(text, encoding='utf-16')
    text = str(text,'UTF-16')
    
    text1 = re.sub('\s+[-]\s+','',text) # word continuation in the next line
    text1 = re.sub('\.+','.',text1)
    text1 = re.sub('\n','', text1)
    text1 = re.sub('“','',text1)
    #print("text1",text1)
    #text1 = re.sub('\\u','',text1)
    return text1
def summarize_data(extracted_data):
    summary = ''
    for  data in extracted_data:      
        
        prompt2 =f"""Context: {data}\n\n Only use provided text fragments above in the context. Summarize only the text fragments provided in the context. \
                Extract significant information from the provided context . \
                Keep max tokens to 1000.\n\n  """          
        try:
                summary_text = openai.Completion.create(prompt=prompt2, temperature=0, model="text-davinci-003",max_tokens=200,api_key = openai_api_key)
                #time.sleep(10)
                print(summary_text['choices'][0]['text'])
                summary = summary +' '+summary_text['choices'][0]['text']
        except Exception as e:
                print("Exception in summary ",e)
    return summary
     

def main():
    st.title("PDF Data Extractor")
    st.write("Upload a PDF file and extract data based on the provided headlines.")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.read())
        extracted_data = extract_data_from_headlines(temp_path)
        st.subheader("Extracted Data")
        summaries = []
        title_dict ={}
        print("length of extracted data",len(extracted_data))
        for  data in extracted_data:
            #st.write(f"{headline}")
            sentences = fragment_text(data)      
            
            for sentence in sentences:
                st.write(sentence)
            st.write("---")
        summary = ''
        MODEL = "text-embedding-ada-002"
        #summary = summarize_data(extracted_data)
        print(summary)
        #f2 = open('summary.txt','w')
        #f2.write(summary)
        #f2.close()
        #query ="water solution to somalia"
        index = pinecone_embeddings.index_pdf_file(data)        
        pinecone_embeddings.index_json_metadata(metadata_filepath,index)
        queries = pinecone_embeddings.create_queries()
        for query in queries:
            answer = pinecone_embeddings.query_index(index,query,MODEL)
            print(answer)

if __name__ == '__main__':    
    main()
