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

def guess_time_remaining(start_time, i, segments_length):
    time_elapsed = time.time() - start_time
    time_per_segment = time_elapsed / (i + 1)
    return f"approximately {int(round((time_per_segment * (segments_length - i - 1)), 0))} seconds remaining..."

def retry_until_successful(function, t):
    try:
        result = function
        return result
    except:
        print(f"Error. Waiting {t} seconds and trying again.")
        time.sleep(t)
        return retry_until_successful(function, t * 2)

system_message = {"role": "system", "content": prompts["system"]}
allowed_context_injection_tokens = options["max_context_injection_tokens"] - len(enc.encode(system_message["content"]))

def formulate_messages(messages, latest_segment):
    if (calculate_costs.calculate_tokens_in_messages(messages) > options["max_context_injection_tokens"]):
        rebuilt_messages = []
        total_tokens = 0
        for (i) in range(len(messages) - 1, len(messages) - (2 * options["context_queue_size"]) - 1, -1):
            if (messages[i]["role"] == ["assistant"]):
                if total_tokens + len(enc.encode(messages[i]["content"])) + len(enc.encode(messages[i-1]["content"])) <= allowed_context_injection_tokens:
                    rebuilt_messages.insert(0, messages[i-1])
                    rebuilt_messages.insert(0, messages[i])
                else:
                    break
        rebuilt_messages.insert(0, system_message)
        messages = rebuilt_messages
    messages.append(
        {
                "role": "user", 
                "content": prompts["translation_prefix"] + latest_segment + prompts["translation_suffix"]
        }
    )
    return messages

# ----- TRANSLATION ----- #

translated_segments = []
messages = [system_message]

segments_length = len(segments)
start_time = time.time()

for i, segment in enumerate(segments):

    messages = formulate_messages(messages, segment)

    completion = retry_until_successful(openai.ChatCompletion.create(
            model=models["translation"],
            messages=messages
    ), 2)
    
    translation = completion.choices[0].message.content

    embedding_response = openai.Embedding.create(
        input = translation,
        model = models["embedding"]
    )

    translated_segments.append({
        "original": segment,
        "translation": translation,
        "embedding": (embedding_response['data'][0]['embedding'])
    })

    messages.append(completion.choices[0].message)

    print("-----")
    print(translation + "\n")
    print(datetime.now().strftime("%H:%M:%S"))
    print(f"{str(i + 1)}/{str(segments_length)}")
    print(guess_time_remaining(start_time, i, segments_length))
    print("-----")

# ----- EDITING ----- #

# ----- WRITING ----- #

# For .txt file
if (output[".txt"]["enabled"]):
    file = open(f'{output["path"]}.txt', "w")
    for i, object in enumerate(translated_segments):
        if (i != 0):
            file.write(f' {object["translation"]}')
        else: file.write(object["translation"])