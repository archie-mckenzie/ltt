# ltt.py
# Large text translator using vector embeddings and GPT-3.5
# Author: Archie McKenzie 
# © 2023, MIT License

# ----- SETUP ----- #

# Get the config file
import json
config = json.load(open('config.json', 'r'))
models = config["models"]
prompts = config["prompts"]
input = config["input"]

# Set up environment variables
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize OpenAI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY") or config["api_keys"]["OPENAI_API_KEY"]

# ./scripts
from scripts import get_costs, read_input

# NLTK
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

# Tiktoken, for tokenizing sentences
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# ----- INITIAL VARIABLES ----- #

text = read_input.read_input(input["path"], input["type"])

def split_into_sentences(text):

    raw_sentences = nltk.sent_tokenize(text)

    # If the program is instructed to split sentences based on double newlines
    if input["options"]["split_on_double_newlines"]: 
        sentences = []
        for sentence in raw_sentences:
            sentences.extend(sentence.split("\n"))
        return sentences
    
    return raw_sentences # Otherwise return the NLTK sentences

sentences = split_into_sentences(text)
sentence_tokens = [len(enc.encode(sentence)) for sentence in sentences]

# ----- CHECKS ----- #

projected_costs = get_costs.estimate_costs(text, models)

print(sentences[0:25])
print(sentence_tokens[0:5])
