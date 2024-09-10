from flask import *
from flask_cors import CORS
import core_functions
from flask_session import Session
from datetime import timedelta
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "abc" 
CORS(app)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=60)
Session(app)

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
        response  = {
             "msg:" : "Please enter a question" 
            }
        return jsonify(response)  
    else:
        if session_id == None:
                print("No session id")
                session_id = core_functions.create_session() 
        response = core_functions.answer_query(question,pmid,session_id)
        


        return jsonify(response)
          
    
@app.route("/deletesession",methods=['POST'])
def delete_session():
    data_front_end = request.get_json() 
    session_id = data_front_end.get('session_id')
    del session[session_id]
    response = {
         "msg" : "Session deleted successfully"
    }
    return jsonify(response)

app.run(host='0.0.0.0', port=80)
