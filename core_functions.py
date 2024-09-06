import requests
from flask import *
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pymilvus import AnnSearchRequest,RRFRanker
import google.generativeai as genai
import os
from pymilvus import connections, MilvusClient,Collection
from datetime import timedelta



def get_ip():
    ip = requests.get('http://checkip.amazonaws.com').text.strip()
    return ip

ip = get_ip()
client = MilvusClient(uri="http://" + ip + ":19530")
connections.connect(host=ip, port="19530")
vector_data_for_all_fields_with_term = Collection(name="vector_data_for_all_fields_with_term")

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
        "Articles" :extract_section(articles)
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
                section = section.replace(":","")
                content = match.group('content').strip()
                data[section] = content if section not in content else "" 
            print(data)    
            for key in data.keys():
              if data[key] != "":
                temp[key] = data[key]          
         

            results.append(temp)        

        return results
 
