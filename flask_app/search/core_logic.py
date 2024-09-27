from flask import *
import re
from sentence_transformers import SentenceTransformer
from pymilvus import AnnSearchRequest,RRFRanker
import google.generativeai as genai
from pymilvus import connections, MilvusClient,Collection
import uuid
import os
from flask_app.search.publication_categories import publication_categories 
import threading
from collections import defaultdict

ip =  os.environ['IP']
client = MilvusClient(uri="http://" + ip + ":19530")
connections.connect(host=ip, port="19530")
vector_data_pmc = Collection(name="vector_data_pmc")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])



def get_data(query):
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = sbert_model.encode([query])
    res = vector_data_pmc.search(
         param={"metric_type": "L2", "params": {}} ,
         data = query_embedding,
         anns_field="vector_data",
         limit=100
        )    
    relavent_articles = []
    for hits in res :
        for hit in hits:
            temp = {}
            temp['id'] = hit.id
            temp['score'] = hit.score
            relavent_articles.append(temp)
    relavent_articles = sorted(relavent_articles, key=lambda x: x['id'])
    ids = [article['id'] for article in relavent_articles]
    articles = client.get(
        collection_name="vector_data_pmc",
        ids=ids
    )
    order_lookup = {item['id']: item['score'] for item in relavent_articles}
    articles = sorted(articles, key=lambda article: order_lookup[article['pmid']])

    for article in articles:
        article.pop('vector_data')
        
    response = {
        "articles" :articles
    }
    return response

def answer_query(question,pmid,session_id):
    context = ''
    if len(session[session_id]['history']) == 0:
        article = client.get(
        collection_name="vector_data_pmc",
        ids=[pmid]
        )  
        context = json.dumps(article[0]['body_content']) 
        context = context
    prompt = context +"\n\n" +  question
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="Think yourself as an research assistant.You will receieve data related to life sciences.Analyze it and answer only if a valid question is asked after that",
        safety_settings="BLOCK_NONE",
    )
    chat_session = model.start_chat(
        history=session[session_id]['history']
    )


    response = chat_session.send_message(prompt,stream=True)
    for chunk in response:
        temp = {
            "session_id" : session_id,
            "answer" : chunk.text
        }
        temp = json.dumps(temp)
        yield temp.encode("utf-8")
    for i in chat_session.history[-2:]:
          temp = {}
          temp["role"] = i.role
          temp["parts"] = [part.text for part in i.parts]
          session[session_id]['history'].append(temp)
          

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

            temp["display"] = section_to_display(temp)

            temp["display"] = section_to_display(temp)
            results.append(temp)        

        return results
 
def create_session():
    session_id = str(uuid.uuid4())
    session[session_id] = {
        "history" : []
    }

    return session_id

def section_to_display(article):
    max_length = 0
    for section in article:
        current_length = len(article[section])
        if current_length > max_length:
            max_length = current_length
            largest_section = section
    return largest_section        

def filter_type(query,filters):
    articles = get_data(query)
    temp = []
    for article in articles['articles']:
        for publication_type in article['publication_type']:
            # print(articles)
            for filter in filters:
                if publication_type in publication_categories[filter] and article not in temp:
                    temp.append(article)
                    break
    articles = {
        "articles" : temp
    }        
    return articles            

def annotate(pmids):
    articles = client.get(
        collection_name="vector_data_pmc",
        ids=pmids
    )  
    data = {}
    for pmid in pmids:
        data[pmid] = []
    for article in articles:
        context = json.dumps(article['abstract_content']) + "\n\n" + json.dumps(article['body_content']) 
        chunk = len(context) // 4
        article_chunks = [context[i:i+chunk] for i in range(0,len(context),chunk)]
        threads = []
        for chunk in article_chunks:
            thread = threading.Thread(target=annotate_api_gemini, args=(article['pmid'],chunk,data))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
    for pmid in data.keys():
        total_count = 0
        data[pmid] = merge_dict(data[pmid])
        if len(data[pmid]) > 0:
            for i in data[pmid].keys():
                # print(i)
                values = sum(list(data[pmid][i].values()))
                total_count = total_count + values
            empty_fields = []    
            for j in data[pmid].keys():
                if len(data[pmid][j]) > 0:
                    data[pmid][j]['annotation_score'] = ( sum(list(data[pmid][j].values())) / total_count ) * 100
                else:
                    empty_fields.append(j)
            for k in empty_fields:
                del data[pmid][k]       
    return data

def annotate_api_gemini(pmid,context,data):
    generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        # system_instruction="Think yourself as an research assistant.You will receieve data related to life sciences.Analyze it and answer only if a valid question is asked after that",
        safety_settings="BLOCK_NONE",
    )
    words = context.split(" ")
    prompt = str(words) +"\n\n" +  "Dump all genes, proteins, diseases,gene ontology, mutation,cellular , variants into a json and also give the count of their occurence in the article.Give response only in json format. Format of json : {'gene': {'word': 'occurence_value'},'protein' : {'word': 'occurence_value'} }.Use the keywords 'gene','disesase','gene ontology','celluar','mutation','protein','varaints' for json.If no terms are found related to these categories return an empty json "
    chat_session = model.start_chat()
    response = chat_session.send_message(prompt)
    temp = {}
    response = json.loads(response.text.replace("```json","").replace("```","").replace("'",'"'))
    data[pmid].append(response)
    return temp

def merge_dict(data):
    merged_dict = {}
    for chunk_response in data:
        for annotate_type in chunk_response.keys():          
            if annotate_type not in merged_dict.keys():
                merged_dict[annotate_type] = chunk_response[annotate_type]
            else:
                for k in chunk_response[annotate_type].keys():
                    flag = False
                    for v in merged_dict[annotate_type].keys():
                        flag = True
                        if k == v:
                            merged_dict[annotate_type][v] = merged_dict[annotate_type][v] + chunk_response[annotate_type][k]  
                    if flag == False:
                        merged_dict[annotate_type][k] = chunk_response[annotate_type][k]  
    return merged_dict

# def dict_to_list(dictionary):
#     data = []
#     for item in dictionary:
#         temp = {}
#         # print(dictionary[item])
#         if isinstance(dictionary[item],dict):
#             for key,value in dictionary[item].items():
#                 temp['title'] = key
#                 if isinstance(value,dict):
#                     # print(value)
#                     temp['content'] = dict_to_list(value)
#                 else:
#                     temp['content'] = value
#                 data.append(temp)
#         else:
#             temp['title'] = item
#             temp['content'] = dictionary[item]
#             data.append(temp)        
#     return data