from pymilvus import connections, MilvusClient,Collection
import re
from flask import *
from flask_cors import CORS
from transformers import pipeline
import core_functions
import uuid
from flask_session import Session
from datetime import timedelta
import google.generativeai as genai
from datetime import timedelta

ip = "13.235.71.25"
print(ip)
genai.configure(api_key="AIzaSyDPCCwRJyLVLzv4QP7jwu8M9aEC87WrNMQ")
client = MilvusClient(uri="http://" + ip + ":19530")
connections.connect(host=ip, port="19530")
vector_data_for_all_fields_with_term = Collection(name="vector_data_for_all_fields_with_term")

app = Flask(__name__)
app.secret_key = "abc" 
CORS(app)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=60)
Session(app)
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

def answer_query(question,session_id,pmid):
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

    data = []
    for i in chat_session.history:
          temp = {}
          temp["role"] = i.role
          temp["parts"] = [part.text for part in i.parts]
          data.append(temp)
    session[session_id]['history'] = data
    print("#########SESSSION########")
    print(session[session_id]['history'] )
    answer = {
        "Answer":response.text,
        "Session_Id" : session_id
    }
    
    return answer

@app.route("/query",methods=['POST'])
def get_results():
    data_front_end = request.get_json() 
    query = data_front_end.get('query')

    if query == "":
        msg  = {"msg:" : "Please enter a query" }
        response = jsonify(msg)
        return response
    else:
            response = core_functions.get_data(query)
            return jsonify(response)
    
@app.route("/generateanswer",methods=['POST'])
def get_answer():
    data_front_end = request.get_json() 
    question = data_front_end.get('question')
    session_id = data_front_end.get('session_id')
    print(session_id)
    pmid = data_front_end.get('pmid')

    if question == "":
        msg  = {"msg:" : "Please enter a question" }
        response = jsonify(msg)
        return response
    else:
            if session_id == None:
                    print("No session id")
                    session_id = str(uuid.uuid4())
                    session[session_id] = {
                        "history" : []
                    }
            # chat_session = core_functions.init_chat_bot(pmid)
            response = answer_query(question,session_id,pmid)
            return jsonify(response)    

app.run(host='0.0.0.0', port=80)
