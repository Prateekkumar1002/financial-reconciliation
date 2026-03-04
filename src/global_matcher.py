import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from .feature_engineering import compute_features
from .config import EMBEDDING_MODEL, MATCH_THRESHOLD, FAISS_TOP_K
from .faiss_index import FAISSIndex

class GlobalMatcher:

    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def match(self, bank, check, model):

        bank_emb = self.embedder.encode(bank.clean_desc.tolist(), convert_to_numpy=True)
        check_emb = self.embedder.encode(check.clean_desc.tolist(), convert_to_numpy=True)

        faiss_index = FAISSIndex(check_emb)
        _, candidate_indices = faiss_index.search(bank_emb, k=FAISS_TOP_K)

        score_matrix = np.zeros((len(bank), len(check)))

        for i in range(len(bank)):
            for j in candidate_indices[i]:
                features = compute_features(
                    bank.iloc[i],
                    check.iloc[j],
                    bank_emb[i],
                    check_emb[j]
                )

                score = model.predict(features.reshape(1, -1))[0]
                score_matrix[i, j] = score

        cost_matrix = 1 - score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for r, c in zip(row_ind, col_ind):
            if score_matrix[r, c] >= MATCH_THRESHOLD:
                matches.append(
                    (bank.iloc[r].transaction_id,
                     check.iloc[c].transaction_id,
                     score_matrix[r, c])
                )

        return matches