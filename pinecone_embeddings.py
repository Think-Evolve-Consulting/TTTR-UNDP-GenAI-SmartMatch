import json
import pandas as pd
import time
import pinecone
import os
import openai
from collections import Counter
from tqdm import tqdm
import tiktoken
import metadata_embeddings
os.OPENAI_API_KEY ="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
metadata_filepath = 'Digital X Solution Catalog Metadata for Hack to the Rescue.xlsx'
def create_pinecone_index(index_name,MODEL):
   
	pinecone.init(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",environment="northamerica-northeast1-gcp",  )    
	res = openai.Embedding.create(input=["Sample document text goes here",], engine=MODEL)
	embeds = [record['embedding'] for record in res['data'][0:1]]
	index = None	
	try:
		if index_name not in pinecone.list_indexes():
			pinecone.create_index(index_name, dimension=len(embeds[0]))
			index = pinecone.Index(index_name)			
		else:
			index = pinecone.Index(index_name)			
	except Exception as e:
		print("Exception while creating pinecone index: ",e)
	return index
def store_embeddings(index,text_list,MODEL):	
	print("Upserting to pinecone index: ",len(text_list))   
	count = 0  # we'll use the count to create unique IDs
	batch_size = 16  # process everything in batches of 32
	vectors = []
	usage = Counter()
	total_emb_cost = 0	
	for i in tqdm(range(0,len(text_list),batch_size)):
		text = text_list[i:i+batch_size]
		print("length of text ",len(text))		
		
		embeds = []
		for i in range(len(text)):
			resp = openai.Embedding.create(
            input=[text[i]], engine=MODEL)
			embeds.append(resp['data'][0]['embedding']) #= [record['embedding'] for record in resp['data'][0]]
		vectors.extend(embeds)
		print("length embeds ",len(embeds))
		# set end position of batch
		i_end = min(i+batch_size, len(text_list))
		print("size ",len(text))
		lines_batch = text 
		ids_batch = [str(n) for n in range(i, i_end)]
		#res = get_vectors(lines_batch)
		print("len ids_batch ",len(ids_batch))		
		print("Length of embeds = ",len(embeds))
		meta = [{'text': line} for line in lines_batch]
		print("len meta ",len(meta))
		to_upsert = []
		for j in range(len(embeds)):
			doc = {'id':ids_batch[j],"values":embeds[j],'metadata':meta[j]}
			to_upsert.append(doc) #zip(ids_batch, embeds, meta)
		# upsert to Pinecone
		vector = []  		
		print("Upsert batch of vectors to pinecone: ",len(list(to_upsert)))
		try:
			index.upsert(vectors=to_upsert)			
		except Exception as e:
			print("Error while upserting: ",e)	
	print("After upserting Pinecone Index stats: ",index.describe_index_stats())			
	return index
tokenizer_model = 'text-davinci-003'
encoding = tiktoken.encoding_for_model(tokenizer_model)
def get_token_count(text):
	return tokenizer_model.token_count(text)
	used += 0.04   * data.get('total_tokens:gpt-4',0) / 1000 # prompt_price=0.03 but output_price=0.06
	used += 0.02   * data.get('total_tokens:text-davinci-003',0) / 1000
	used += 0.002  * data.get('total_tokens:text-curie-001',0) / 1000
	used += 0.002  * data.get('total_tokens:gpt-3.5-turbo',0) / 1000
	used += 0.0004 * data.get('total_tokens:text-embedding-ada-002',0) / 1000
	return used
def query_by_vector(vector, index, limit=None):
	"return (ids, distances and texts) sorted by cosine distance"	
	res = index.query(
                      vector=vector,
                      top_k=5,
                      include_metadata=True
                      )
	
	resp = res['matches']	
	resp.sort(key = lambda x:x['score'],reverse=True)	
	id_list = []
	dist_list = []
	text_list = []
	for i in range(len(resp)):
		id_list.append(int(resp[i]['id']))
		dist_list.append(resp[i]['score'])
		text_list.append(resp[i]['metadata']['text'])	
	return id_list, dist_list, text_list
def query_index(index,query,MODEL):
	print("Query: ", query)	
	resp = openai.Embedding.create(input=[query], engine=MODEL)	
	v = resp['data'][0]['embedding']	
	id_list, dist_list, text_list = query_by_vector(v, index, limit=5)
	selected = {} 
	selected2 = text_list	
	SEPARATOR = '\n---\n'
	context = ''
	context_len = 0
	frag_list = []
	for text in selected2:
		frag = text
		frag_len = len(text)
		if context_len+frag_len <= 3000: 
			context += SEPARATOR + frag 
			frag_list += [frag]
			#context_len = get_token_count(context)
	task = "Get sigificant information from the context based on given question. "
	prompt = f"""
		{task}		
		Context:
		{context}		
		Question: {query}		
		Answer:""" 
	
	# GET ANSWER	
	resp = openai.Completion.create(prompt=prompt, temperature=0, model="text-davinci-003",max_tokens=400,api_key = openai.api_key)
	answer = resp['choices'][0]['text']
	return answer
	
def index_pdf_file(data):
    MODEL = "text-embedding-ada-002"
    index_name = 'undp-index'
    index = create_pinecone_index(index_name,MODEL)
    index = store_embeddings(index,data,MODEL)    
    return index
def index_json_metadata(metadata_filepath,index):
	MODEL = "text-embedding-ada-002"
	index_name = 'undp-index'
	json_data = metadata_embeddings.get_json_data(metadata_filepath)		
	index = store_embeddings(index,json_data,MODEL) 
def create_queries():
	sdg_list = {"sdg 17": "partnerships for the goals", "sdg 12": "responsible consumption and production", "sdg 13": "climate action",\
	      "sdg 11": "sustainable cities and communities", "sdg 5": "gender equality", "sdg 7": "affordable and clean energy", \
	    "sdg 1": "no poverty", "sdg 4": "quality education", "sdg 3": "good health and well-being", "sdg 10": "reduced inequalities", \
	    "sdg 16": "peace, justice and strong institutions", "sdg 9": "industry, innovation, and infrastructure", \
	     "sdg 8": "decent work and economic growth", "sdg 2": "zero hunger", "sdg 6": "clean water and sanitation", "sdg 15": "life on land"}
	
	queries = []
	#for key in sdg_list.keys():
	queries.append('what are the key requirements for the development goals mentioned in mentioned in the document?')
	queries.append("Give the details of finance in USD given to the project mentioned in the document.")
	#queries.append('what is the project about and what is the SDG mentioned in the text?')
	return queries
		
		
	
	

			
