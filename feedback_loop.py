# feedback_loop.py
import csv
from pathlib import Path
FEEDBACK_FILE = Path("feedback_corrections.csv")

def flag_and_record(sample_text, predicted_label, user_label):
    # append to CSV for later retraining
    header = ["merchant", "predicted", "correct_label"]
    exists = FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([sample_text, predicted_label, user_label])
    print("Recorded correction to", FEEDBACK_FILE)

if __name__ == "__main__":
    # example: interactive correction
    s = input("Enter merchant text (example: 'AMZ Mktp'):\n")
    # import model to predict
    import joblib, json
    model = joblib.load("artifacts/model_pipeline.joblib")
    pred = model.predict([s])[0]
    prob = model.predict_proba([s]).max()
    print(f"Model predicted: {pred} (conf={prob:.2f})")
    if prob < 0.60:
        print("Low confidence — please enter correct label from config.json categories.")
        correct = input("Correct label:\n")
        flag_and_record(s, pred, correct)
    else:
        ans = input("Is prediction correct? (y/n): ")
        if ans.strip().lower().startswith('n'):
            correct = input("Correct label:\n")
            flag_and_record(s, pred, correct)
        else:
            print("No correction recorded.")
