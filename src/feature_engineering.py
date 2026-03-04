import numpy as np
from math import exp
from sklearn.metrics.pairwise import cosine_similarity
from .config import SIGMA_AMOUNT, SIGMA_DAYS

def compute_features(b, c, b_emb, c_emb):

    cosine_sim = cosine_similarity(
        b_emb.reshape(1, -1),
        c_emb.reshape(1, -1)
    )[0][0]

    text_score = (cosine_sim + 1) / 2
    amount_score = exp(-abs(b.amount - c.amount) / SIGMA_AMOUNT)
    date_score = exp(-abs((b.date - c.date).days) / SIGMA_DAYS)
    type_score = 1 if b.norm_type == c.norm_type else 0

    return np.array([text_score, amount_score, date_score, type_score])