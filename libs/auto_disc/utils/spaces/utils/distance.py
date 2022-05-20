"""
    all distance function
"""

def calc_l2(embedding_a, embedding_b):
    dist = (embedding_a - embedding_b).pow(2).sum(tuple(range(1,embedding_b.ndim))).sqrt()
    return dist