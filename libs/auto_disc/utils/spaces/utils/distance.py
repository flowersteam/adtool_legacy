"""
    all distance function
"""

def calc_l2(embedding_a, embedding_b):
    dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()
    return dist