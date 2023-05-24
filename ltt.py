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
from scripts import get_metrics, read_input, segment_text

# ----- READ THE TEXT ----- #

text = read_input.read_input(input["path"], input["type"])
sentences = nltk.sent_tokenize(text)
segments = segment_text.segment(sentences, options)
segment_tokens = [len(enc.encode(segment)) for segment in segments]

# ----- CHECKS ----- #

print(sentences[0:2])
print(segments[0:2])
print(segment_tokens[0:2])

if (get_metrics.get_final_approval(segments, models, options)):
    print()
else:
    print("Halted! No action has been taken")
    exit()

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

# ----- TIME ----- #

start_time = time.time()
    
# ----- EMBEDDING ----- #

if (models["embedding"]):
    embeddings = []
    print("----- START OF SEGMENT EMBEDDING -----") 
    for i, segment in enumerate(segments):
        if len(advanced["embedding_kwargs"]) == 0:
            embeddings.append(retry_until_successful(2, openai.Embedding.create, model=models["embedding"], input=segment)["data"][0]["embedding"])
        else:
            embeddings.append(retry_until_successful(2, openai.Embedding.create, model=models["embedding"], input=segment, **advanced["embedding_kwargs"])["data"][0]["embedding"])
        print(f"Embedded segment {i + 1}/{len(segments)}")
    print("----- END OF SEGMENT EMBEDDING -----")

# ----- TRANSLATION ----- #

translated_segments = []

translation_time = time.time()

print("----- START OF TRANSLATION -----")
for i, segment in enumerate(segments):
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": formulate_prompt(i)}
    ]
    if len(advanced["translation_kwargs"]) == 0:
        translation = retry_until_successful(2, openai.ChatCompletion.create, model=models["translation"], messages=messages)["choices"][0]["message"]["content"]
    else:
        translation = retry_until_successful(2, openai.ChatCompletion.create, model=models["translation"], messages=messages, **advanced["translation_kwargs"])["choices"][0]["message"]["content"]
    translated_segments.append(translation)
    print("-----")
    print(translation + "\n")
    print(datetime.now().strftime("%H:%M:%S"))
    print(f"{i + 1}/{len(segments)}")
    print(guess_time_remaining(translation_time, i, len(segments)))
    print("-----")
print("----- END OF TRANSLATION -----")

# ----- EDITING ----- #

paragraphs = []

editing_time = time.time()

# Feed many translated segments at once into it (as many as the limit allows), ask it to use double newlines to denote new paragraphs
# Split the model output along double newlines and store it in paragraphs
# Integrate the last paragraph into the next prompt so that all inter-segment gaps are joined
if (models["editing"]):
    print("----- START OF EDITING -----")
    i = 0
    current_prompt = f'{prompts["editing"]}\n\n'
    current_tokens = 0
    while i <= len(translated_segments) - 1:
        print(f"{paragraphs}, {i}")
        if len(paragraphs) > 1:
            current_prompt += f'{paragraphs[len(paragraphs) - 1]}\n\n'
            current_tokens += len(enc.encode(paragraphs[len(paragraphs) - 1]))
            paragraphs = paragraphs[:-1]
        while current_tokens + len(enc.encode(translated_segments[i])) <= options["max_editing_tokens"] - len(enc.encode(prompts["editing"])):
            print(f'Current tokens: {current_tokens}')
            print(f'Max tokens: {options["max_editing_tokens"] - len(enc.encode(prompts["editing"]))}')
            if (current_prompt[len(current_prompt) - 1]) == "\n":
                current_prompt += translated_segments[i]
            else: 
                current_prompt += f' {translated_segments[i]}'
            current_tokens += len(enc.encode(translated_segments[i]))
            i += 1
            if i >= len(translated_segments) - 1: break
        
        print(f'Current tokens: {current_tokens}')
        print("-----")
        print(current_prompt)
        print("-----")
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": current_prompt}
        ]
        if (len(advanced["editing_kwargs"])) == 0:
            edit = retry_until_successful(2, openai.ChatCompletion.create, model=models["translation"], messages=messages)["choices"][0]["message"]["content"]
        else:
            edit = retry_until_successful(2, openai.ChatCompletion.create, model=models["translation"], messages=messages, **advanced["editing_kwargs"])["choices"][0]["message"]["content"]
        for j, paragraph in enumerate(edit.split('\n\n')):
            if (j != 0): print(f'\n{paragraph}')
            else: print(paragraph)
            paragraphs.append(paragraph)
        current_prompt = f'{prompts["editing"]}\n\n'
        current_tokens = 0
        print("-----")
        print(datetime.now().strftime("%H:%M:%S"))
        print(f"{i}/{len(segments)}")
        print(guess_time_remaining(editing_time, i - 1, len(translated_segments)))
        print("-----")
    print("----- END OF EDITING -----")
else: paragraphs = translated_segments

print(f"Total time elapsed: {get_metrics.convert_to_human_time(time.time() - start_time)}")

# ----- WRITING ----- #

# For .txt file
if (output[".txt"]["enabled"]):
    file = open(f'{output["path"]}{output[".txt"]["name"]}.txt', "w")
    separator = "\n\n" if models["editing"] else " "
    for i, paragraph in enumerate(paragraphs):
        if (i != 0): file.write(f'{separator}{paragraph}')
        else: file.write(paragraph)

# For .pdf file
# Write a separate script

# For metadata json
if (output[".json"]["enabled"]):
    metadata = {
        "segments": [],
        "paragraphs": []
    }
    for i, segment in segments:
        metadata["segments"].append({
            "original": segment,
            "translation": translated_segments[i],
            "embedding": embeddings[i]
        })
    for paragraph in paragraphs:
        metadata["paragraphs"].append(paragraph)
    with open(f'{output["path"]}{output[".json"]["name"]}.json', "w") as json_file:
        json.dump(metadata, json_file)