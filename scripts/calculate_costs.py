# estimate_costs.py
# Author: Archie McKenzie 
# © 2023, MIT License

# ----- IMPORTS ----- #

# Tiktoken, for tokenizing sentences
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# ----- CALCULATE COSTS ----- #

def calculate_tokens_in_messages(messages):
    total = 0
    for message in messages:
        total += len(enc.encode(message["content"]))
    return total

# ----- ESTIMATE COSTS ----- #

def estimate_costs(segments, models, options):

    total_costs = 0

    cost_menu = {
        "gpt-4-32k": (0.12 / 1000),
        "gpt-4": (0.06 / 1000),
        "gpt-3.5-turbo": (0.002 / 1000),
        "text-embedding-ada-002": (0.0004 / 1000)
    }

    print("\nEstimating costs...")

    # Estimating Costs for Translation
    try:
        translation_costs = 0
        for segment in segments:
            translation_costs += (cost_menu[models["translation"]] * (len(enc.encode(segment)) + options["max_context_injection_tokens"]) + (cost_menu[models["translation"]] * len(enc.encode(segment))))
        translation_costs = round(translation_costs, 3)
    except KeyError:
        translation_costs = 0
    print(f"Translation: ${translation_costs}")
    total_costs += translation_costs
    
    # Estimating Costs for Embedding
    try:
        embedding_costs = 0
        for segment in segments:
            embedding_costs += (cost_menu[models["embedding"]] * len(enc.encode(segment)))
        embedding_costs = round(embedding_costs, 3)
    except KeyError:
       embedding_costs = 0
    print(f"Embedding: ${embedding_costs}")
    total_costs += embedding_costs

    # Estimating Costs for Editing
    try:
        editing_costs = 0
        for segment in segments:
            editing_costs += (cost_menu[models["editing"]] * 2 * len(enc.encode(segment)))
        editing_costs = round(editing_costs, 3)
    except KeyError:
        editing_costs = 0
    print(f"Editing: ${editing_costs}")
    total_costs += editing_costs

    print(f"Total Cost: ${total_costs}")
    if input("Approve? (y/n) ") == 'n':
        print("Halted! No action has been taken")
        exit()
    
    print()
    return total_costs
