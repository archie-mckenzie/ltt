# segment_text.py
# Author: Archie McKenzie 
# © 2023, MIT License

# ----- IMPORTS ----- #

# Tiktoken, for tokenizing sentences
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Regular Expressions, for NLP
import re

# ----- FUNCTIONS ----- #

# For the edge case in which divide() fails to split the string enough
def split_string(string, max_length):
    words = string.split(' ')
    result = []
    current = words[0]
    for word in words[1:]:
        if len(word) < 1: continue
        if len(current) + len(word) + 1 > max_length:  # +1 for the space
            if len(current) > max_length: # For the edgier case in which there is a word longer than max_length
                for sequence in [current[i:i+max_length] for i in range(0, len(current), max_length)]:
                    result.append(sequence)
            else:
                result.append(current)
            current = word
        else: current += ' ' + word
    if current:
        if len(current) > max_length: # For the edgier case in which there is a word longer than max_length
            for sequence in [current[i:i+max_length] for i in range(0, len(current), max_length)]:
                result.append(sequence)
        else: result.append(current)
    return result

# Divides on newlines in order to meet max_segment_tokens limit
def divide(segment, options):
    division = re.split(r'\n\s*\n', segment)
    division_tokens = [len(enc.encode(part)) for part in division]
    result = []
    print(division_tokens)
    for i, part in enumerate(division):
        if division_tokens[i] < 1: continue
        if division_tokens[i] > options["max_segment_tokens"]:
            for fragment in (split_string(part.strip() + '\n\n', options["max_segment_tokens"])):
                result.append(fragment)
        else:
            result.append(part.strip() + '\n\n')
    return result

# Joins together sentences into longer segments which are then translated
# Takes an array of strings (sentences), returns an array of strings (segments)
# Also takes input options from config.json
def segment(sentences, options):

    sentence_tokens = [len(enc.encode(sentence)) for sentence in sentences]
    
    segments = []
    current_segment = ""
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        if (current_tokens < options["split_token_threshold"]):
            if current_segment == "": 
                current_segment = sentence
                current_tokens += sentence_tokens[i]
            else:
                current_segment += f" {sentence}"
                current_tokens += sentence_tokens[i]
        if (current_tokens >= options["split_token_threshold"]):
            if (current_tokens > options["max_segment_tokens"]):
                segment_fragments = segment(divide(current_segment, options), options)
                for fragment in segment_fragments:
                    segments.append(fragment)
                current_segment = ""
                current_tokens = 0
            else:
                segments.append(current_segment)
                current_segment = ""
                current_tokens = 0
    
    if (options["split_on_double_newlines"]):
        segments = re.split(r'\n\n', '\n'.join(segments))
    
    return segments
