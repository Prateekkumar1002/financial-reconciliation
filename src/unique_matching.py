def unique_matching(bank, check):
    matches = []
    matched_bank = set()
    matched_check = set()

    bank_counts = bank["amount"].value_counts()
    check_counts = check["amount"].value_counts()

    unique_amounts = set(
        bank_counts[bank_counts == 1].index
    ).intersection(
        set(check_counts[check_counts == 1].index)
    )

    for amt in unique_amounts:
        b = bank[bank["amount"] == amt].iloc[0]
        c = check[check["amount"] == amt].iloc[0]

        matches.append((b.transaction_id, c.transaction_id, 1.0))
        matched_bank.add(b.transaction_id)
        matched_check.add(c.transaction_id)

    return matches, matched_bank, matched_check