"""
    all distance function
"""

def calc_l2(embedding_a, embedding_b):
    # L2 + add regularizer to avoid dead outcomes
    dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt() - 0.5 * embedding_b.pow(2).sum(-1).sqrt()
    return dist