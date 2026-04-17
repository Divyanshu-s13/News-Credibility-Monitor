import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

from src.config.config import MODEL_DIR


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(
        y_test, y_pred, target_names=["Real", "Fake"], output_dict=True
    )

    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}  (weighted)")
    print(f"  Recall    : {recall:.4f}  (weighted)")
    print(f"  F1 Score  : {f1:.4f}  (weighted)")
    print("\nPer-Class Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print("Confusion Matrix:\n", cm)

    metrics = {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "per_class": {
            cls: {k: round(v, 4) for k, v in vals.items()}
            for cls, vals in report.items()
            if cls not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plot_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved → {plot_path}")

    return metrics
