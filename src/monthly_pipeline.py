import numpy as np
import random

from .evaluation import evaluate
from .unique_matching import unique_matching
from .global_matcher import GlobalMatcher
from .similarity_model import SimilarityModel
from .feature_engineering import compute_features


class MonthlyPipeline:

    def __init__(self, bank, check):
        self.bank = bank.reset_index(drop=True)
        self.check = check.reset_index(drop=True)

    # ============================================================
    # Utility: Extract numeric part of transaction ID
    # ============================================================

    def extract_numeric_id(self, tid):
        return ''.join(filter(str.isdigit, str(tid)))

    # ============================================================
    # Build Balanced Training Dataset
    # ============================================================

    def build_training_data(self, embedder):

        print("Building training dataset...")

        bank_emb = embedder.encode(
            self.bank.clean_desc.tolist(),
            convert_to_numpy=True
        )

        check_emb = embedder.encode(
            self.check.clean_desc.tolist(),
            convert_to_numpy=True
        )

        X = []
        y = []

        # Build numeric ID mapping for check dataset
        check_numeric_map = {}
        for idx, row in self.check.iterrows():
            numeric_id = self.extract_numeric_id(row.transaction_id)
            check_numeric_map[numeric_id] = idx

        # ===============================
        # Positive Samples (True Matches)
        # ===============================

        for i in range(len(self.bank)):
            b = self.bank.iloc[i]
            numeric_id = self.extract_numeric_id(b.transaction_id)

            if numeric_id not in check_numeric_map:
                continue  # skip if no matching check transaction

            true_check_idx = check_numeric_map[numeric_id]
            c = self.check.iloc[true_check_idx]

            features = compute_features(
                b,
                c,
                bank_emb[i],
                check_emb[true_check_idx]
            )

            X.append(features)
            y.append(1)

        # ===============================
        # Negative Samples (Random Mismatch)
        # ===============================

        for i in range(len(self.bank)):
            b = self.bank.iloc[i]
            b_numeric = self.extract_numeric_id(b.transaction_id)

            # Sample random check transaction that is NOT correct match
            while True:
                random_idx = random.randint(0, len(self.check) - 1)
                c = self.check.iloc[random_idx]
                c_numeric = self.extract_numeric_id(c.transaction_id)

                if c_numeric != b_numeric:
                    break

            features = compute_features(
                b,
                c,
                bank_emb[i],
                check_emb[random_idx]
            )

            X.append(features)
            y.append(0)

        print(f"Training samples: {len(X)} (balanced classes)")

        return np.array(X), np.array(y)

    # ============================================================
    # Human Validation Simulation
    # ============================================================

    def simulate_human_validation(self, matches):
        """
        For synthetic dataset:
        True match if numeric ID matches.
        """
        validated = []

        for b_id, c_id, score in matches:
            if self.extract_numeric_id(b_id) == self.extract_numeric_id(c_id):
                validated.append((b_id, c_id, score))

        return validated

    # ============================================================
    # Main Monthly Pipeline
    # ============================================================

    def run(self):

        print("\nStep 1: Unique Matching")
        unique_matches, matched_bank_ids, matched_check_ids = unique_matching(
            self.bank,
            self.check
        )

        print(f"Unique Matches Found: {len(unique_matches)}")

    # -------------------------------------------------
    # Remove matched rows BEFORE ML stage
    # -------------------------------------------------

        unmatched_bank = self.bank[
            ~self.bank.transaction_id.isin(matched_bank_ids)
        ].reset_index(drop=True)

        unmatched_check = self.check[
            ~self.check.transaction_id.isin(matched_check_ids)
        ].reset_index(drop=True)

        print(f"Unmatched Transactions: {len(unmatched_bank)}")

        if len(unmatched_bank) == 0:
            print("All transactions matched via unique amounts.")

            evaluate(unique_matches, len(self.bank))
            return unique_matches

        print("\nStep 2: Training Similarity Model")

        matcher = GlobalMatcher()
        embedder = matcher.embedder

    # IMPORTANT: train on FULL dataset (for better learning)
        X, y = self.build_training_data(embedder)

        model = SimilarityModel()
        model.train(X, y)

        print("Model training completed.")

        print("\nStep 3: Global Matching (Only Unmatched Rows)")

        ml_matches = matcher.match(unmatched_bank, unmatched_check, model)

        print(f"ML Matches Found: {len(ml_matches)}")

    # Combine matches
        all_matches = unique_matches + ml_matches

        print("\nStep 4: Human Validation Simulation")

        validated_matches = self.simulate_human_validation(all_matches)

        evaluate(validated_matches, len(self.bank))

        return validated_matches