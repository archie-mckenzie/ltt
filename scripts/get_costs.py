# estimate_costs.py
# Author: Archie McKenzie 
# © 2023, MIT License

# Import tiktoken, for calculating OpenAI model costs
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def estimate_costs(text, models):

    total_costs = 0

    cost_menu = {
        "gpt-4": 0.06 / 1000,
        "gpt-3.5-turbo": 0.002 / 1000,
        "text-embedding-ada-002": 0.0004 / 1000
    }

    text_length = len(enc.encode(text)) # Calculate the number of tokens

    # Translation
    try:
        translation_costs = round(cost_menu[models["translation"]] * text_length, 3)
    except KeyError:
        translation_costs = 0
    print(f"Translation: ${translation_costs}")
    total_costs += translation_costs

    # Embedding
    try:
        embedding_costs = round(cost_menu[models["embedding"]] * text_length, 3)
    except KeyError:
        embedding_costs = 0
    print(f"Embedding: ${embedding_costs}")
    total_costs += embedding_costs

    # Analysis costs
    try:
        analysis_costs = round(cost_menu[models["analysis"]] * text_length, 3)
    except KeyError:
        analysis_costs = 0
    print(f"Analysis: ${analysis_costs}")
    total_costs += analysis_costs

    # Translation costs
    try:
        editing_costs = round(cost_menu[models["editing"]] * text_length, 3)
    except KeyError:
        editing_costs = 0
    print(f"Editing: ${editing_costs}")
    total_costs += editing_costs

    print(f"Total: ${total_costs}")
    if input("Approve? (y/n) ") == 'n':
        print("Halted! No action has been taken")
        exit()
    
    return total_costs
