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
options = input["options"]
output = config["output"]
advanced = config["advanced"]

# Set up environment variables
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize OpenAI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY") or config["api_keys"]["OPENAI_API_KEY"]

# Tiktoken, for tokenizing sentences
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

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

# Time
import time
from datetime import datetime

# NumPy, for calculating similarity between vectors
import numpy as np

# ./scripts
from scripts import calculate_costs, read_input, segment_text

# ----- READ THE TEXT ----- #

text = read_input.read_input(input["path"], input["type"])
sentences = nltk.sent_tokenize(text)
segments = segment_text.segment(sentences, options)
segment_tokens = [len(enc.encode(segment)) for segment in segments]

# ----- CHECKS ----- #

print(sentences[0:2])
print(segments[0:2])
print(segment_tokens[0:2])

projected_costs = calculate_costs.estimate_costs(segments, models, options)

# ----- TRANSLATION HELPER FUNCTIONS ----- #

allowed_context_injection_tokens = options["max_context_injection_tokens"] - len(enc.encode(prompts["system"])) - len(enc.encode(prompts["initial"]))

def guess_time_remaining(start_time, i, segments_length):
    time_elapsed = time.time() - start_time
    time_per_segment = time_elapsed / (i + 1)
    return f"approximately {int(round((time_per_segment * (segments_length - i - 1)), 0))} seconds remaining..."

def retry_until_successful(t, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except:
        print(f"Error! Waiting for {t} seconds then trying again.")
        time.sleep(t)
        return retry_until_successful(t * 2, function, *args, **kwargs)

def formulate_prompt(i):

    # Translation Prompt
    prompt = f'{prompts["translation_prefix"]}\n\n{segments[i]}\n\n{prompts["translation_suffix"]}'

    included_indeces = []
    total_tokens = 0
 
    # Queue-Structure Context Injection
    for j in range(i - 1, max(i - options["context_queue_size"] - 1, -1), -1):
        current_tokens = len(enc.encode(segments[j]))
        if total_tokens + current_tokens <= allowed_context_injection_tokens:
            included_indeces.insert(0, j)
            total_tokens += current_tokens
    if (len(included_indeces) > 0):
        prompt = f'{prompts["queue_structured_context_injection"]}\n\n{" ".join([segments[j] for j in included_indeces])}\n\n{prompt}'

    # Embedding-Based Context Injection
    if (models["embedding"]):
        similarities = []
        alpha = np.array(embeddings[i])
        for j in range(i):
            beta = np.array(embeddings[j]) 
            # Calculate the dot product of the two vectors
            dot_product = np.dot(alpha, beta)
            # Calculate the norm (length) of each vector
            alpha_norm = np.linalg.norm(alpha)
            beta_norm = np.linalg.norm(beta)
            # Calculate the cosine similarity between the two vectors
            similarities.append(dot_product / (alpha_norm * beta_norm))
        if (len(similarities) > 0):
            threshold = np.mean(similarities) + (advanced["outlier_threshold"] * np.std(similarities))
            champion_indeces = []
            if threshold >= advanced["minimum_similarity"]:
                for j, similarity in enumerate(similarities):
                    if similarity >= threshold and j not in included_indeces:
                        current_tokens = len(enc.encode(segments[j]))
                        if total_tokens + current_tokens <= allowed_context_injection_tokens:
                            champion_indeces.insert(0, j)
                            total_tokens += current_tokens
                if (len(champion_indeces) > 0):
                    prompt = f'{prompts["embedding_similarity_context_injection"]}\n\n{" ".join([segments[j] for j in champion_indeces])}\n\n{prompt}'
    
    return prompt if prompts["initial"] == "" else f'{prompts["initial"]}\n\n{prompt}'

    
# ----- EMBEDDING ----- #

embeddings = []

print("----- START OF SEGMENT EMBEDDING -----")
for i, segment in enumerate(segments):
    embeddings.append(retry_until_successful(2, openai.Embedding.create, model=models["embedding"], input=segment)["data"][0]["embedding"])
    print(f"Embedded segment {i + 1}/{len(segments)}")
print("----- END OF SEGMENT EMBEDDING -----")

# ----- TRANSLATION ----- #

translated_segments = []

start_time = time.time()

for i, segment in enumerate(segments):
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": formulate_prompt(i)}
    ]
    translation = retry_until_successful(2, openai.ChatCompletion.create, model=models["translation"], messages=messages)["choices"][0]["message"]["content"]
    translated_segments.append(translation)
    print("-----")
    print(translation + "\n")
    print(datetime.now().strftime("%H:%M:%S"))
    print(f"{str(i + 1)}/{str(len(segments))}")
    print(guess_time_remaining(start_time, i, len(segments)))
    print("-----")

# ----- EDITING ----- #

finalized_segments = []

# Loop through translated segments
# At least two at a time, potentially cover the same segment twice
# So that all inter-segment gaps are joined

# ----- WRITING ----- #

# For .txt file
if (output[".txt"]["enabled"]):
    file = open(f'{output["path"]}.txt', "w")
    for i, segment in enumerate(finalized_segments):
        if (i != 0):
            file.write(f' {segment}')
        else: file.write(segment)

# For .pdf file

# For metadata.json