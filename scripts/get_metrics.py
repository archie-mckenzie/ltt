# estimate_costs.py
# Author: Archie McKenzie 
# © 2023, MIT License

# ----- IMPORTS ----- #

# Tiktoken, for tokenizing sentences
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# ----- GET FINAL APPROVAL ----- #

def get_final_approval(segments, models, options):
    print(f"Total Cost: ${estimate_costs(segments, models, options)}")
    print(f"Estimated Time: {estimate_timing(segments, models, options)}")
    if input("Approve? (y/n) ") == 'n':
        return False
    return True

# ----- ESTIMATE COSTS ----- #

cost_menu = {
    "gpt-4-32k": (0.12 / 1000),
    "gpt-4": (0.06 / 1000),
    "gpt-3.5-turbo": (0.002 / 1000),
    "text-embedding-ada-002": (0.0004 / 1000)
}

def estimate_costs(segments, models, options):

    total_costs = 0

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

    return round(total_costs, 3)

# ----- ESTIMATE TIMING ----- #

def convert_to_human_time(t):
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    return f"{hours}h {minutes}m {seconds}s"

def estimate_timing(segments, models, options):

    t = 0

    timing_menu = { # empirically determined in testing
        "gpt-4-32k": (39 / 1000),
        "gpt-4": (39 / 1000),
        "gpt-3.5-turbo": (22 / 1000), 
        "text-embedding-ada-002": (0.4 / 100)
    }

    # Estimating Timing for Translation
    try:
        for segment in segments:
            t += (timing_menu[models["translation"]] * (len(enc.encode(segment)) + options["max_context_injection_tokens"]))
    except KeyError: pass

    # Estimating Timing for Embedding
    try:
        for segment in segments:
            t += (timing_menu[models["embedding"]] * len(enc.encode(segment)))
    except KeyError: pass

    # Estimating Timing for Editing
    try:
        for segment in segments:
            t += (timing_menu[models["editing"]] * len(enc.encode(segment)))
    except KeyError: pass

    return convert_to_human_time(t)

#