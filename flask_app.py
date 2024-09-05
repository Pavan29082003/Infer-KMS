from pymilvus import connections, MilvusClient,Collection
import re
from flask import Flask,request,jsonify
from flask_cors import CORS
from transformers import pipeline
import core_functions

ip = core_functions.get_ip()
print(ip)
client = MilvusClient(uri="http://" + ip + ":19530")
connections.connect(host=ip, port="19530")
vector_data_for_all_fields_with_term = Collection(name="vector_data_for_all_fields_with_term")

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
            response = core_functions.get_data(query)
            return jsonify(response)
    
@app.route("/generateanswer",methods=['POST'])
def get_answer():
    data_front_end = request.get_json() 
    question = data_front_end.get('question')
    pmid = data_front_end.get('pmid')

    if question == "":
        msg  = {"msg:" : "Please enter a question" }
        response = jsonify(msg)
        return response
    else:
            # chat_session = core_functions.init_chat_bot(pmid)
            response = core_functions.answer_query(pmid,question)
            return jsonify(response)    

app.run(host='0.0.0.0', port=80)
