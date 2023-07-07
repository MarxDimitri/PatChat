from langchain.chat_models import ChatOpenAI
from config import openai_key
from config import elasticsearch_username, elasticsearch_password, elasticsearch_endpoint, elasticsearch_index_name
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import SequentialChain
import json
import streamlit as st

st.set_page_config(
    page_icon=":robot:",
    page_title="PatChat"
)
st.title("🔎 PatChat")

os.environ['OPENAI_API_KEY'] = openai_key
llm = OpenAI(temperature=0.6)

# Create the HuggingFace Transformer like before
model_name = "sentence-transformers/all-distilroberta-v1"
hf = HuggingFaceEmbeddings(model_name=model_name)

# Connect with Elasticsearch
es_endpoint = elasticsearch_endpoint
es_username = elasticsearch_username
es_password = elasticsearch_password
es_index_name = elasticsearch_index_name

es_url = f"https://{es_username}:{es_password}@{es_endpoint}:9243"
es_db = ElasticVectorSearch(embedding=hf, elasticsearch_url=es_url, index_name=es_index_name)


def step_1_identify_key_concepts_chain():
    key_concepts_question = """
       You are a scientist or patent layer.
       It is your job to find prior art in patents.
       The following context describes patents consisting of title, abstract, publication number and some other information: {context}
       Identify the key innovations of the patent. Describe them in few sentences.
       When providing an answer, prefix with publication number followed by title.
       Ignore any citation publication numbers.
       When asked: {question}
       Your answer to the questions using only information in the context is: """
    key_concepts_prompt = PromptTemplate(template=key_concepts_question, input_variables=["context", "question"])
    return LLMChain(prompt=key_concepts_prompt, llm=llm, output_key="key_concepts")


def step_2_identify_keywords_chain():
    keywords_question = """
        Given the key concepts of a patent, generate keywords for further prior art research.
        Use synonyms and related keywords based on your knowledge.
        Here are the identified key concepts: {key_concepts}
    """
    keywords_prompt = PromptTemplate(input_variables=['key_concepts'], template=keywords_question)
    return LLMChain(prompt=keywords_prompt, llm=llm, output_key="keywords")


def step_3_determine_ipc_symbols():
    # use previously generated keywords to identify patents and aggregate their IPC sympbols
    return "H01L 31/00"


chain = SequentialChain(
    chains=[
        step_1_identify_key_concepts_chain(),
        step_2_identify_keywords_chain()
    ],
    input_variables=['question', 'context'],
    output_variables=['key_concepts', "keywords"],
    verbose=False
)


def ask_question(question):
    # get the relevant chunk from Elasticsearch for a question
    similar_docs = es_db.similarity_search(question)
    context = similar_docs[0].page_content
    return chain({"question": question, "context": context})


with st.sidebar:
    "[Dashboards](https://patchat.kb.europe-west3.gcp.cloud.es.io:9243/app/dashboards#/view/dad0d6e0-257e-11ed-9ee3-7f2ce5c4cf8b?_g=(filters:!()))"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can help with your research."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Put your question here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key, streaming=True)

    with st.chat_message("assistant"):
        response = ask_question(prompt)
        output = response['key_concepts'] + '\n' + response['keywords']
        output = output.replace("Publication number:", "Publication number:\n")
        output = output.replace("Title:", "Title:\n")
        output = output.replace("Key concepts:", "\n\nKey concepts:\n")

        output.replace("Answer:", "")
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.write(output)
