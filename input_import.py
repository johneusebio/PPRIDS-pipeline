# importing data
import pandas as pd

def interpret_input(filepath:str, keywords:list):
    """Interpret the input of a textfile containing parameters. Each line represents a unique datapoint with different attributes of data denoted by the keywords. The file is formatted as:
    
    KEYWORD_1=[content here], KEYWORD_2=[content here], KEYWORD_3=[content here]
    KEYWORD_1=[content here], KEYWORD_2=[content here], KEYWORD_3=[content here]
    KEYWORD_1=[content here], KEYWORD_2=[content here], KEYWORD_3=[content here]

    Args:
        filepath (str): filepath to the input text file
        keywords (list): list of keywords (str) to search for within the text file
    """
    
    input_data = open(filepath, "r")
    lines = input_data.readlines()

    input_df = pd.DataFrame(index=range(len(lines)), columns=keywords)

    for line, ind in zip(lines, range(len(lines))):
        line=line.strip()
        input_dict=interpret_input_line(line, keywords)

        for kw in keywords:
            input_df.loc[ind, kw] = input_dict[kw]

    return(input_df)

def interpret_input_line(line:str, keywords:list):
    """Interprets an individual line for the interpret_input function

    Args:
        line (str): line from the input text file
        keywords (list): list of keywords (str) to earch for within the text file
    """
    input_dict = {}
    items = line.split(",")
    
    for item in items:
        item=item.strip()
        try:
            key, val = parse_input(item, keywords)
            input_dict[key] = val
        except:
            print("Error:", item)
    return(input_dict)

def parse_input(text:str, keywords:list, split:str="=", first:str="[", last:str="]"):
    """Parsing the line from the input text file's line into interpretable data.

    Args:
        text (str): The line being parsed
        keywords (list): list of keywords (str) to search for within the text file
        split (str, optional): Character used to split the keywords from the values. Defaults to "=".
        first (str, optional): Opening character denoting the values. Defaults to "[".
        last (str, optional): Terminating character denoting the values. Defaults to "]".
    """
    key, val = parse_line(text, keywords, split)
    val = strip_chars(val, first, last)
    return(key, val)

def strip_chars(text:str, first:str, last:str):
    """Strips the provided text from flanking characters

    Args:
        text (str): Text being stripped
        first (str): Opening flank character to strip
        last (str): Closing flank character to strip
    """
    text = f"{text[1:] if text.startswith(first) else text}"
    text = f"{text[:-1] if text.endswith(last) else text}"
    return(text)

def parse_line(text:str, keywords:list, split:str="="):
    """Splits line into keyword and accompanying value

    Args:
        text (str): text file to split
        keywords (list): keyword
        split (str, optional): Character to split along, KEYWORD=[content]. Defaults to "=".

    Raises:
        Exception: If the text doesn't contain the specified keyword.
    """
    key, val = text.split(split)
    if key not in keywords:
        raise Exception("Input doesn't contain any of the specified key words")
    return(key, val)

def check_exist(filepath:str, type="any"):
    """Checks if a file or directory exists

    Args:
        filepath (str): Filepath to check the existence of
        type (str, optional): Check for the existence of a file ("file", "f") or directory ("dir", "d"). Defaults to "any".

    Raises:
        Exception: Invalid type is entered (neither a file or directory).
    """
    if type in ["file", "f"]:
        from os.path import isfile as exists
    elif type in ["dir", "d"]:
        from os.path import isdir as exists
    elif type=="any":
        from os.path import exists
    else:
        raise Exception("type '" + type + "' is invalid.")
    return(exists(filepath))

def txt2list(filepath:str):
    with open(filepath) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines