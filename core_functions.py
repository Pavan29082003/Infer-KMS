import requests
from flask import *
import re
from sentence_transformers import SentenceTransformer
from pymilvus import AnnSearchRequest,RRFRanker
import google.generativeai as genai
from pymilvus import connections, MilvusClient,Collection
from datetime import timedelta
import uuid

def get_ip():
    ip = requests.get('http://checkip.amazonaws.com').text.strip()
    return ip

ip = get_ip()
client = MilvusClient(uri="http://" + ip + ":19530")
connections.connect(host=ip, port="19530")
vector_data_for_all_fields_with_term = Collection(name="vector_data_for_all_fields_with_term")

genai.configure(api_key="AIzaSyDPCCwRJyLVLzv4QP7jwu8M9aEC87WrNMQ")
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }
model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction="You are a research assistant"
    )

def get_data(query):
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = sbert_model.encode([query])
    ids = vector_data_for_all_fields_with_term.hybrid_search(
            reqs=[
                AnnSearchRequest(
                    data=[query_embedding[0]],  
                    anns_field='VECTOR_SEARCH_TERM',  
                    param={"metric_action": "L2"}, 
                    limit=50
                ),
                AnnSearchRequest(
                    data=[query_embedding[0]],  
                    anns_field='VECTOR_DATA',  
                    param={"metric_action": "L2"}, 
                    limit=50
                )         
            ],
            rerank=RRFRanker(), 
            limit=25
        )    
    ids = re.findall(r'id: (\d+)', str(ids[0]))
    articles = client.get(
        collection_name="vector_data_for_all_fields_with_term",
        ids=ids
    )
    response = {
        "articles" :extract_section(articles)
    }
    return response

def answer_query(question,pmid,session_id):
    context = ''
    if len(session[session_id]['history']) == 0:
        article = client.get(
        collection_name="vector_data_for_all_fields_with_term",
        ids=[pmid]
        )   
        context = article[0].get('TEXT_DATA')
    prompt = context + question

    chat_session = model.start_chat(
        history=session[session_id]['history']
    )
    response = chat_session.send_message(prompt)
    for i in chat_session.history[-2:]:
          temp = {}
          temp["role"] = i.role
          temp["parts"] = [part.text for part in i.parts]
          session[session_id]['history'].append(temp)

    print(session[session_id]['history'] )
    response = {
        "answer":response.text,
        "session_Id" : session_id
    }
    
    return response

def extract_section(articles):
        results = []
        for article in articles:
            temp = {}
            temp['PMID'] = article.get('PMID')
            abstract = article.get('TEXT_DATA')
            pattern = r"(?P<section>\b[A-Z][A-Za-z]{3,}):(?P<content>.*?)(?=\b[A-Z][A-Za-z]{3,}:|$)"
            matches = re.finditer(pattern, abstract, re.DOTALL)
            data = {}
            for match in matches:
                section = match.group('section')
                content = match.group('content').strip()
                data[section] = content if section not in content else "" 
            print(data)    
            for key in data.keys():
              if data[key] != "":
                temp[key] = data[key]          
         

            results.append(temp)        

        return results
 
def create_session():
    session_id = str(uuid.uuid4())
    session[session_id] = {
        "history" : []
    }

    return session_id
