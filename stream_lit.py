import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from pymilvus import connections, MilvusClient
import json
import re

client = MilvusClient(uri="http://52.66.241.51:19530")
connections.connect(host='52.66.241.51', port='19530')


def semantic_search(query, type):
    # Use a more efficient model for embeddings
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = sbert_model.encode([query])
    res = client.search(
        collection_name="vector_data_for_all_fields_with_term", 
        data=[query_embedding[0]],  # No need to convert to tensor since no GPU
        limit=10,
        anns_field =  "VECTOR_SEARCH_TERM",
        search_params={"metric_type": "L2",}
    )
    res = json.dumps(res[0])
    res = json.loads(res)
    res = client.get(
        collection_name="vector_data_for_all_fields_with_term",
        ids=[article["id"] for article in res]
    )
    if type == "qna":
        return [response['VECTOR_DATA'] for response in res]
    return res

# Adjusted title with larger font size
st.markdown("<h1 style='font-size: 48px;'>Medical Research Assistant</h1>", unsafe_allow_html=True)
st.write("Enter your query below:")

query = st.text_input("Search Query")

if query.strip() == "":
    st.write("Please enter a search query.")
else:
    with st.spinner('Processing your request...'):
        if query.endswith("?"):
            def answer_query(query):
                relevant_texts = semantic_search(query, "qna")
                model_name = "google/flan-t5-base"  # Smaller model
                tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)

                # HuggingFacePipeline and LLMChain initialized once
                flan_llm = HuggingFacePipeline(pipeline=pipe)
                prompt_template = ChatPromptTemplate.from_template("Answer for the following question:\n\n{question}")
                llm_chain = LLMChain(llm=flan_llm, prompt=prompt_template)
                if relevant_texts:
                    context = relevant_texts[0]
                    prompt = f"{context}\nQuestion: {query}\nAnswer:"
                    return llm_chain.run({"question": prompt})
                else:
                    return "No relevant information found."

            response = answer_query(query)
            st.write(f"Answer: {response}")

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

            for article in articles:
                pmid = article.get('PMID')
                abstract = article.get('TEXT_DATA')
                data = extract_section(abstract)
                title = data.get("TITLE")
                introduction = data.get("INTRODUCTION")
                methods = data.get("METHODS")
                results_findings = data.get("RESULTS")
                conclusion = data.get("CONCLUSION")
                keywords = data.get("KEYWORDS")
                search_term = data.get("SEARCHTERM")
                if pmid:
                    st.write(f"Pmid: {pmid}")
                if title:
                    st.markdown(f"<h2 <strongstyle='font-size: 32px;'>{title}</strong></h2>", unsafe_allow_html=True)
                if introduction:
                    st.write(f"Purpose/Background: {introduction}")
                if methods:
                    st.write(f"Methods: {methods}")
                if results_findings:
                    st.write(f"Results/Findings: {results_findings}")
                if conclusion:
                    st.write(f"Conclusion: {conclusion}")
                if keywords:
                    st.write(f"Keywords: {keywords}")
                if search_term:
                    st.write(f"Search Term: {search_term}")
                st.write("=" * 80)