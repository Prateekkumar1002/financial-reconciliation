def extract_numeric_id(tid):
    return ''.join(filter(str.isdigit, str(tid)))


def evaluate(matches, total):

    correct = 0

    for b_id, c_id, _ in matches:
        if extract_numeric_id(b_id) == extract_numeric_id(c_id):
            correct += 1

    predicted = len(matches)

    precision = correct / predicted if predicted else 0
    recall = correct / total
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n===== Evaluation =====")
    print(f"Matches: {predicted}")
    print(f"Correct: {correct}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("======================\n")