# ltt.py
# Large text translator using vector embeddings and GPT-3.5
# Author: Archie McKenzie 
# © 2023, MIT License

# ----- SETUP ----- #

# Get the config file
import json
config = json.load(open('config.json', 'r'))
input = config["input"]

# Set up environment variables
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize OpenAI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY") or config["api_keys"]["OPENAI_API_KEY"]

# ./scripts
from scripts import read_input

# ----- INITIAL VARIABLES ----- #

text = read_input.read_input(input["path"], input["type"])

print(text)

# ----- CHECKS ----- #

# Calculate the number of tokens

