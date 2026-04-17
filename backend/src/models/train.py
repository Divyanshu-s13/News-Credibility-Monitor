from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Corrects for Real/Fake class imbalance in ISOT dataset
        C=1.0,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model