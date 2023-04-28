from langchain import HuggingFaceHub
from getpass import getpass
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
model_name = "climatebert/distilroberta-base-climate-f"

import tabula

# Read pdf into list of DataFrame
dfs = tabula.read_pdf("pages\Water_pond_tanks_2021.pdf", pages='all')
display(dfs)