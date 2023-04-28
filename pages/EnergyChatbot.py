from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
import os
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from langchain import PromptTemplate
from streamlit_chat import message
st.set_page_config(layout="wide")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(model_name="gpt-3.5-turbo")

reader = PdfReader("pages\\61628_BK_EERE-EnergySavers_w150.pdf")
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
def query(prompt):
    docs = docsearch.similarity_search(prompt)
    return chain.run(input_documents=docs, question=prompt)

texts = text_splitter.split_text(raw_text)
docsearch = FAISS.from_texts(texts, embeddings)
chain = load_qa_chain(OpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff")

st.markdown("<h1 style='text-align: center; color: black; font-weight: bold;'>Energy Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: black; font-style: italic;'>Interact with our chatbot to find out about how you can save more energy in your day to day life and more!</h4>", unsafe_allow_html=True)





if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text 


user_input = get_text()

if user_input:
    output = query(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
