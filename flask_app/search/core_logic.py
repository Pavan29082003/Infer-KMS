from datetime import datetime, timedelta
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
vector_data_biorxiv = Collection(name="vector_data_biorxiv")
vector_data_plos = Collection(name="vector_data_plos")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def get_data(query):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = sbert_model.encode([query])
    res_pmc = vector_data_pmc.search(
         param={"metric_type": "COSINE", "params": {}} ,
         data = query_embedding,
         anns_field="vector_data",
         limit=100
        )
    res_biorxiv = vector_data_biorxiv.search(
         param={"metric_type": "COSINE", "params": {}} ,
         data = query_embedding,
         anns_field="vector_data",
         limit=100
        )         
    res_plos = vector_data_plos.search(
         param={"metric_type": "COSINE", "params": {}} ,
         data = query_embedding,
         anns_field="vector_data",
         limit=100
        )          
    relavent_articles = []
    for hits_pmc,hits_biorxiv,hits_plos in zip(res_pmc,res_biorxiv,res_plos) :
        for hit_pmc,hit_biorxiv,hit_plos in zip(hits_pmc,hits_biorxiv,hits_plos):
            temp1 = {}
            temp1['id'] = hit_pmc.id
            temp1['score'] = hit_pmc.score
            temp2 = {}
            temp2['id'] = hit_biorxiv.id
            temp2['score'] = hit_biorxiv.score 
            temp3 = {}
            temp3['score'] = hit_plos.score       
            temp3['id'] = hit_plos.id         
            relavent_articles.append(temp1)
            relavent_articles.append(temp2)
            relavent_articles.append(temp3)
    relavent_articles = sorted(relavent_articles, key=lambda x: x['score'], reverse= True)
    ids = [article['id'] for article in relavent_articles]
    articles_pmc = client.get(
        collection_name="vector_data_pmc",
        ids=ids
    )
    articles_biorxiv = client.get(
        collection_name="vector_data_biorxiv",
        ids=ids
    )   
    articles_plos = client.get(
        collection_name="vector_data_plos",
        ids=ids
    )      
    articles = []

    for article_biorxiv, article_pmc, articles_plos in zip(articles_biorxiv, articles_pmc, articles_plos):
        articles.append(article_biorxiv)
        articles.append(article_pmc)
        articles.append(articles_plos)
    for article in articles:
        print(type(article))
    id_names = {
        "pubmed" : "pmid",
        "BioRxiv" : "bioRxiv_id",
        "Public Library of Science (PLOS)" : "plos_id"
    }
    order_lookup = {item['id']: item['score'] for item in relavent_articles}
    articles = sorted(articles, key=lambda article: order_lookup[article['pmid'] if article.get('pmid') else article[id_names[article['source']]]], reverse=True)

    for article in articles:
        article['similarity_score'] = (  ( order_lookup[article['pmid'] if article.get('pmid') else article[id_names[article['source']]]] + 1 ) / 2 ) * 100
        article.pop('vector_data')
        
    response = {
        "articles" :articles
    }
    return response

def answer_query(question,id,session_id,source):
    context = ''
    collections  = {
        "pubmed" : "vector_data_pmc",
        "biorxiv" : "vector_data_biorxiv",
        "plos": "vector_data_plos"
    }
    print(collections[source])
    if len(session[session_id]['history']) == 0:
        article = client.get(
        collection_name=collections[source],
        ids=[id]
        )  
        context = json.dumps(article[0]['body_content'])  + json.dumps(article[0]['abstract_content'])
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

def annotate(**sources_ids):
    data = {}
    collections  = {
        "pubmed" : "vector_data_pmc",
        "biorxiv" : "vector_data_biorxiv",
        "plos" : "vector_data_plos"
    }
    articles = []
    for source,ids in sources_ids.items():
        if ids:
            articles = articles + client.get(
                collection_name=collections[source],
                ids=ids
            )
            for id in ids:
                data[id] = []
    response = []
    id_names = {
        "pubmed" : "pmid",
        "BioRxiv" : "bioRxiv_id",
        "Public Library of Science (PLOS)" : "plos_id"
    }
    for article in articles:
        source = article['source'] if article.get('source') else "pubmed"
        context = json.dumps(article['abstract_content']) + "\n\n" + json.dumps(article['body_content']) 
        chunk = len(context) // 20
        article_chunks = [context[i:i+chunk] for i in range(0,len(context),chunk)]
        threads = []
        for chunk in article_chunks:
            thread = threading.Thread(target=annotate_api_gemini, args=(article[id_names[source]],chunk,data))
            threads.append(thread)
            thread.daemon = True
            thread.start()
        for thread in threads:
            thread.join()
    for id in data.keys():
        total_count = 0
        data[id] = merge_dict(data[id])
        if len(data[id]) > 0:
            for i in data[id].keys():
                # print(i)
                values = sum(list(data[id][i].values()))
                total_count = total_count + values
            empty_fields = []    
            for j in data[id].keys():
                if len(data[id][j]) > 0:
                    data[id][j]['annotation_score'] = ( sum(list(data[id][j].values())) / total_count ) * 100
                else:
                    empty_fields.append(j)
            for k in empty_fields:
                del data[id][k]       
        response.append({id:data[id]})    
    return response

def annotate_api_gemini(id,context,data):
    try: 
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
        # print(response.text)
        response = json.loads(response.text.replace("```json","").replace("```","").replace("'",'"'))
        data[id].append(response)
    except Exception as e:
        print(e)  
        print(data[id]) 
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
                        if k == v:
                            flag = True
                            merged_dict[annotate_type][v] = merged_dict[annotate_type][v] + chunk_response[annotate_type][k]  
                    if flag == False:
                        merged_dict[annotate_type][k] = chunk_response[annotate_type][k]
    return merged_dict

def filterByDate(pmids,filter_type,from_date,to_date):
    today = datetime.today()
    
    if filter_type == "1 year":
        date_from = today - timedelta(days=365)
        date_to = today
    elif filter_type == "5 years":
        date_from = today - timedelta(days=5 * 365)
        date_to = today
    elif filter_type == "10 years":
        date_from = today - timedelta(days=10 * 365)
        date_to = today
    elif filter_type == "Custom Range" and from_date and to_date:
        date_from = datetime.strptime(from_date, "%d-%m-%Y")
        date_to = datetime.strptime(to_date, "%d-%m-%Y")
    else:
        raise ValueError("Invalid filter type or custom dates not provided.")

    filtered_articles = []
    articles = client.get(
        collection_name="vector_data_pmc",
        ids=pmids
    )  
    print(date_from)
    print(date_to)
    months = {
        'Jan' : "01",
        'Feb' : "02",
        'Mar' : "03",
        'Apr' : "04",
        'May' : "05",
        'Jun' : "06",
        'Jul' : "07",
        'Aug' : "08",
        'Sep' : "09",
        'Oct' : "10",
        'Nov' : "11",
        'Dec' : "12",

     }
    for article in articles:
        article.pop('vector_data')
        pub_date = article.get('publication_date', None)
        pub_date = pub_date.split("-")
        pub_date = str(pub_date[0]) + "-" + str(months[pub_date[1]]) +"-"+ str(pub_date[2])
        if pub_date:
            pub_date_dt = datetime.strptime(pub_date, "%d-%m-%Y")
            if date_from <= pub_date_dt <= date_to:
                filtered_articles.append(article)
    response = {
        "articles" : filtered_articles
    }

    return response

