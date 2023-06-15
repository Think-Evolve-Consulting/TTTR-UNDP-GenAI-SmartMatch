import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import openai
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import os
os.OPENAI_API_KEY ="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
openai_api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
metadata_filepath = 'Digital X Solution Catalog Metadata for Hack to the Rescue.xlsx'
def display_json(json_str):
  for key in json.loads(json_str).keys():
    print(key,':',json.loads(json_str)[key])
catalog_data = pd.read_excel(metadata_filepath) 
#list_of_jsons = []
#print(catalog_data.head())
#for row in catalog_data.iterrows():
   #row_json_obj = pd.DataFrame(row).to_json(orient = 'records')
   #print(row_json_obj)
   #list_of_jsons.append(row_json_obj)
list_of_jsons = catalog_data.to_json(orient='records', lines=True).splitlines()
#display_json(list_of_jsons[0])
#time.sleep(20)
json_data_strings = []
for json_obj in list_of_jsons:
   json_str = json.dumps(json_obj)
   json_str = json_str.replace('\\','')
   json_data_strings.append(json_str)
print("No of json strings ",len(json_data_strings))
f = open("sample.txt",'w')
for jstr in json_data_strings:
   f.write(jstr)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                os.OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
chroma_client = chromadb.Client()

# Add the content vectors
embeds = []
embeddings = OpenAIEmbeddings(openai_api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

total_pages = []
for doc in json_data_strings:
   f = open("file.txt",'w')
   f.write(doc)
   f.close()
   loader = TextLoader("file.txt")
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=0)
   documents = loader.load()
   texts = text_splitter.split_documents(documents)
   
   total_pages.extend(texts)   
pages_doc = total_pages
#vectordb = Chroma.from_documents(pages_doc, embeddings)
metadata_collection = chroma_client.create_collection(name='catalog_metadata', embedding_function=openai_ef)
#vectordb.add_documents(pages_doc)
#vectordb.add_documents(total_pages)
#pdf_qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",max_tokens = 200,openai_api_key = "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"), vectordb, top_k_docs_for_context = 5,\
                                    #return_source_documents=True) 
def query_collection(collection, query, max_results):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances']) 
    print("*********************",results)
    df = pd.DataFrame({
                'id':results['ids'][0], 
                'score':results['distances'][0],   
                           
                })
    
    return df
try:
   ids = []
   for i in range(1,catalog_data.shape[0]):
       ids.append(str(i))
   metadata_collection.add(
         ids=ids,
         embeddings=[pages for pages in pages_doc],    
         )
      
except Exception as e:
   print("Exception while creating an embedding ",e)
q = "Big data and analytics of global diseases using predictive platforms to steer the future of public health."
query_result = query_collection(
    collection=metadata_collection,
    query=q,
    max_results=2, 
    
)
print("\n\n-------------------------------------------------------------------------------")
print(query_result)
query_result.to_csv("result.xlsx")

