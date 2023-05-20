# translate.py
# Large text translator using vector embeddings and GPT-3.5
# Author: Archie McKenzie 
# © 2023, MIT License

# ----- SETUP ----- #

# Get the config file
import json
config = json.load(open('config.json', 'r'))

# Load the environment variables from .env
from dotenv import load_dotenv
import os
load_dotenv()


import openai
openai.api_key = os.getenv("API_KEY") or config.api_keys.OPENAI_API_KEY

# ----- CHECKS ----- #

# Calculate the number of tokens

