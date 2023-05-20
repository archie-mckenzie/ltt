# read_input.py
# Author: Archie McKenzie 
# © 2023, MIT License

def read_input(path, type):
    match type:
        case ".txt":
            return open(path + type, 'r').read()