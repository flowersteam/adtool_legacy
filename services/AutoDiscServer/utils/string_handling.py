import re

def get_word_end_pos(txt, word):
    return txt.index(word) + len(word) - 1

def get_first_number(txt):
    return re.search(r'\d+', txt).group()

def get_number_after_substring(txt, word):
    newstart = get_word_end_pos(txt, word)
    return get_first_number(txt[newstart:])

def get_numbers_in_string(txt, list_key):
    numbers_dict = dict.fromkeys(list_key, None)
    for key in numbers_dict:
        numbers_dict[key] = get_number_after_substring(txt, key)
    return numbers_dict

def match_except_number(txt1, txt2):
    return (
        [i for i in txt1 if not i.isdigit()] 
        == [i for i in txt2 if not i.isdigit()]
    )