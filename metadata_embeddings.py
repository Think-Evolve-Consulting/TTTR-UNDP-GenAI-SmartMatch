import json
import pandas as pd

import openai
from langchain.llms import OpenAI
import openai_functions
from langchain.document_loaders import TextLoader
import time
import os
os.OPENAI_API_KEY ="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"
openai_api_key = "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"
metadata_filepath = 'Digital X Solution Catalog Metadata for Hack to the Rescue.xlsx'
def display_json(json_str):
  for key in json.loads(json_str).keys():
    print(key,':',json.loads(json_str)[key])
def get_json_data(metadata_filepath):
   catalog_data = pd.read_excel(metadata_filepath) 
   list_of_jsons = catalog_data.to_json(orient='records', lines=True).splitlines()
   json_data_strings = []
   for json_obj in list_of_jsons:
      json_str = json.dumps(json_obj)
      json_str = json_str.replace('\\','')
      json_data_strings.append(json_str)
   print("No of json strings ",len(json_data_strings))
   f = open("sample.txt",'w')
   for jstr in json_data_strings:
      f.write(jstr)


   total_pages = []
   for doc in json_data_strings:
      f = open("file.txt",'w')
      f.write(doc)
      f.close()
      loader = TextLoader("file.txt")
      pages = loader.load()
      total_pages.extend(pages)   
      pages_doc = pages
   return json_data_strings



