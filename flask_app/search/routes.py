from flask import *
from flask import Blueprint
from flask_app.search import core_logic

search= Blueprint('search', __name__)

@search.route("/query",methods=['POST'])
def get_results():
    data_front_end = request.get_json() 
    query = data_front_end.get('query')
    if query == "":
        msg  = {"msg:" : "Please enter a query" }
        response = jsonify(msg)
        return response
    else:
            response = core_logic.get_data(query)
            return jsonify(response)
    
@search.route("/filter",methods=['POST'])
def filter_data():
    data_front_end = request.get_json() 
    query = data_front_end.get('query')
    filters = data_front_end.get('filters')
    response = core_logic.filter_type(query,filters)

    return jsonify(response)    

@search.route("/generateanswer",methods=['POST'])
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
                session_id = core_logic.create_session() 

        return Response(stream_with_context(core_logic.answer_query(question,pmid,session_id)),content_type="application/json")
          
    
@search.route("/deletesession",methods=['POST'])
def delete_session():
    data_front_end = request.get_json() 
    session_id = data_front_end.get('session_id')
    del session[session_id]
    response = {
         "msg" : "Session deleted successfully"
    }
    return jsonify(response)


@search.route("/annotate",methods=['POST'])
def annotate():
    data_front_end = request.get_json() 
    pmid = data_front_end.get('pmid')
    response = core_logic.annotate(pmid)
    return jsonify(response)

@search.route("/filterdate",methods=['POST'])
def filterdate():
    data_front_end = request.get_json() 
    pmids = data_front_end.get('pmids')
    filter_type = data_front_end.get('filter_type')
    from_date = data_front_end.get('from_date')
    to_date = data_front_end.get('to_date')
    response = core_logic.filterByDate(pmids,filter_type,from_date,to_date)
    return jsonify(response)