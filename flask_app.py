from pymilvus import connections, MilvusClient,Collection
import re
from flask import Flask,request,jsonify
from flask_cors import CORS
from transformers import pipeline
import core_functions

ip = core_functions.get_ip()
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

app.run(host='0.0.0.0', port=80)
