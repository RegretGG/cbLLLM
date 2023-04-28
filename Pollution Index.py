import os
import streamlit as st
from PIL import Image

from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from langchain import PromptTemplate
os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(model_name="gpt-3.5-turbo")
st.markdown("<h1 style='text-align: center; color: red;'>How safe is your city?</h1>", unsafe_allow_html=True)
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
City = st.text_input(label="Enter your city.")
Age = st.text_input(label="Enter your age.")
Medical = st.text_input(label="Enter any medical condition you have.")
p = 0
if len(City) != 0 and len(Age)!=0 and len(Medical)!= 0:
    template ="How is the air quality in {A}"
    prompt_template = PromptTemplate(
        input_variables=["A"],
        template=template
    )
    agent = create_csv_agent(OpenAI(temperature=0), 'a.csv')
    p = agent.run(prompt_template.format(A= City))
    template1 = """Answer the question based on the context below. If the
    question cannot be answered using the information provided answer
    with "I don't know".


    Air Quality Index(AQI)#(Pollution level) 	 Possible HealthConsequences                                                                                                                                          Advice for General Population	                                                              Advice for Vulnerable Population 	                                                                                                                                                                                                    
                                                                            

    Good
    (0-50) 	                    Low Risk for general and vulnerable.                                                                                                                                                  No special precautions	                                                                  No special precautions
    
    Satisfactory
    (51-100)                    Minor breathing discomfort in vulnerable population / Low risk for general population.                                                                                                No special precautions                                                                      Reduce prolonged or strenuous outdoor physical exertion
                                

    Moderate(101-200)           Low risk for general population / Breathing or other health realted discomfort in vulnerable poplution                                                                               Reduce prolonged or strenuous outdoor physical exertion                                     Avoid prolonged or strenuous outdoor physical exertion

    Poor
    (201-300)                   Breathing discomfort in general population on prolonged exposure / Breathing or other health realted discomfort in vulnerable population on short exposure                            Avoid outdoor physical exertion                                                             Avoid outdoor physical activities

    Very Poor
    (301-400)                   Respiratory illness in general population on prolonged exposure / Pronounced respiratory or other illnesses in vulnerable population on short exposure                                Avoid outdoor physical activities, especially during morning and late evening hours         Remain indoors and keep activity level low

    Severe
    (401-500)                  Respiratory illness in general population on prolonged exposure / Serious respiratory or other illnesses in vulnerable population on short exposure                                    Avoid outdoor physical activities                                                           Remain indoors and keep activity level low



    Vulnerable population (high risk): Elderly, children under 5 years, pregnant women, and ONLY respiratory and cardiovascular diseases. Other ilnesses are not to be considered and are general population.


    {a}

    I live here. 
    My age is {b}.
    The medical conditions I have are: {c}
    How is the air quality here? What is the AQIValue here? What are the possible health consequences I will face based on my population category? What do you advice me to do based on my population category?

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["a","b","c"],
        template=template1
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(llm(prompt_template.format(a=p, b = Age, c= Medical)))
    image = Image.open('Screenshot 2023-04-28 110507.png')
    st.image(image)
