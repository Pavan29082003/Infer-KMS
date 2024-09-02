from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from pymilvus import connections, MilvusClient,Collection,AnnSearchRequest,RRFRanker
import re
from flask import Flask,request,jsonify
from flask_cors import CORS
from transformers import pipeline
import requests

ip = requests.get('http://checkip.amazonaws.com').text.strip()
print(ip)
print(type(ip))
client = MilvusClient(uri=str("http://"+ip+":19530"))
connections.connect(host=ip, port="19530")
vector_data_for_all_fields_with_term = Collection(name="vector_data_for_all_fields_with_term")
model_directory = "/root/Infer-KMS/google-flan-t5-trained-model"
tokenizer = T5Tokenizer.from_pretrained(model_directory)
model = T5ForConditionalGeneration.from_pretrained(model_directory)
def classify_query(query):
    classifier = pipeline("text-classification", model="distilbert/distilbert-base-uncased")
    query_type = classifier(query)
    return query_type  
 
def semantic_search(query, action):
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = sbert_model.encode([query])
    res = vector_data_for_all_fields_with_term.hybrid_search(
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
    ids = re.findall(r'id: (\d+)', str(res[0]))
    print(ids)
    res = client.get(
        collection_name="vector_data_for_all_fields_with_term",
        ids=ids
    )
    query_type = classify_query(query)
    if query_type['label'] == "LABEL_1":
        context = ''
        for article in res:
            context = context + article["TEXT_DATA"]
        return context
    return res



app = Flask(__name__)
CORS(app)


@app.route("/query",methods=['POST'])
def get_results():
    data_front_end = request.get_json() 
    query = data_front_end.get('query')

    if query == "":
        msg  = {"msg:" : "Please enter a query" }
        response = jsonify(msg)
        return response
    else:
            if classify_query(query) == 0:
                def answer_query(query):
                    relevant_texts = semantic_search(query, "qna")

                    if relevant_texts:
                        input_text = f"question: {query} context: {' '.join(relevant_texts)}"
                        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                        outputs = model.generate(**inputs, max_new_tokens=300)
                        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        return decoded_output
                    else:
                        answer = {
                            "Answer" : "No relevant information found."
                        }
                        return jsonify(answer)

                response = jsonify({"Answer:" : answer_query(query)})          
                return response

            else:
                articles = semantic_search(query, "search")

                def extract_section(text):
                    pattern = r"(?P<section>[A-Za-z]+):(?P<content>.*?)(?=[A-Za-z]+:|$)"
                    matches = re.finditer(pattern, text, re.DOTALL)
                    sections = {}
                    for match in matches:
                        section = match.group('section')
                        content = match.group('content').strip()
                        sections[section] = content if section not in content else ""
                    return sections
                results = []
                for article in articles:
                    temp = {}
                    temp['PMID'] = article.get('PMID')
                    abstract = article.get('TEXT_DATA')
                    data = extract_section(abstract)
                    temp['TITLE'] = data.get("TITLE")
                    temp['INTRODUCTION'] = data.get("INTRODUCTION")
                    temp['METHODS'] = data.get("METHODS")                                                                       
                    temp['RESULTS'] = data.get("RESULTS")
                    temp['CONCLUSION'] = data.get("CONCLUSION")
                    temp['KEYWORDS'] = data.get("KEYWORDS")
                    temp['SEARCHTERM'] = data.get("SEARCHTERM")
                    results.append(temp)

                response = jsonify({"Articles" : results})
                return response

app.run(host='0.0.0.0', port=80)
