import joblib
from sklearn.model_selection import train_test_split

from src.data.load_data import load_and_merge_data
from src.utils.text_cleaner import clean_text
from src.features.build_features import build_vectorizer
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.config.config import MODEL_PATH, VECTORIZER_PATH


def run_training():
    print("Loading data...")
    data = load_and_merge_data()

    print("Cleaning text...")
    data["clean_text"] = data["text"].apply(clean_text)

    print("Building features...")
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(data["clean_text"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Training complete!")


if __name__ == "__main__":
    run_training()