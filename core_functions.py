import requests
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pymilvus import AnnSearchRequest,RRFRanker
import google.generativeai as genai
import os
from pymilvus import connections, MilvusClient,Collection

client = MilvusClient(uri="http://3.109.213.141:19530")
connections.connect(host="3.109.213.141", port="19530")
vector_data_for_all_fields_with_term = Collection(name="vector_data_for_all_fields_with_term")

def get_ip():
    ip = requests.get('http://checkip.amazonaws.com').text.strip()
    return ip

def classify_query(query):
    classifier = pipeline("text-classification", model="shahrukhx01/question-vs-statement-classifier")
    query_type = classifier(query)
    label = {
        "LABEL_0" : "text",
        "LABEL_1" : "question"
    }
    return label[query_type[0]['label']]

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
                )             
            ],
            rerank=RRFRanker(), 
            limit=10
        )    
    ids = re.findall(r'id: (\d+)', str(ids[0]))
    articles = client.get(
        collection_name="vector_data_for_all_fields_with_term",
        ids=ids
    )
    query_type = classify_query(query)
    if query_type == "question":
        prompt = ""
        for article in articles:
            prompt = prompt + article["TEXT_DATA"]
        prompt = prompt + query
        response = {
            "Answer" : answer_query(prompt)
        }
        return response
    response = {
        "Articles" :extract_section(article)
    }
    return response

def answer_query(prompt):
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
    )

    chat_session = model.start_chat(
    history=[]
    )
    print(chat_session)
    response = chat_session.send_message(prompt)
    answer = response.text
    
    return answer


def extract_section(articles):
        results = []
        for article in articles:
            temp = {}
            temp['PMID'] = article.get('PMID')
            abstract = article.get('TEXT_DATA')
            pattern = r"(?P<section>[A-Za-z]+):(?P<content>.*?)(?=[A-Za-z]+:|$)"
            matches = re.finditer(pattern, abstract, re.DOTALL)
            data = {}
            for match in matches:
                section = match.group('section')
                content = match.group('content').strip()
                data[section] = content if section not in content else ""            
            temp['TITLE'] = data.get("TITLE")
            temp['INTRODUCTION'] = data.get("INTRODUCTION")
            temp['METHODS'] = data.get("METHODS")                                                                       
            temp['RESULTS'] = data.get("RESULTS")
            temp['CONCLUSION'] = data.get("CONCLUSION")
            temp['KEYWORDS'] = data.get("KEYWORDS")
            temp['SEARCHTERM'] = data.get("SEARCHTERM")
            results.append(temp)        

        return results
 