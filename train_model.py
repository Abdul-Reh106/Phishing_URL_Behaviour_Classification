from pathlib import Path
from urllib.parse import urlsplit
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay,RocCurveDisplay,accuracy_score,classification_report,confusion_matrix,roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from features import extract_features
from prepare_data import load_clean_data

# Load the new dataset after prefix cleanup and duplicate removal.
data = load_clean_data()

# Skip URLs that cannot be parsed, keeping URLs and labels aligned.
feature_rows = []
valid_indices = []
for index, url in enumerate(data["url"]):
    try:
        feature_rows.append(extract_features(url))
        valid_indices.append(index)
    except ValueError:
        continue

print("\nUnparseable URLs skipped:", len(data) - len(valid_indices))
data = data.iloc[valid_indices].reset_index(drop=True)
X = pd.DataFrame(feature_rows)
y = data["label"]

print("Feature table size:", X.shape)
print(X.head())

print("\nLabel counts:")
print(y.value_counts().sort_index())

# Keep URLs from the same hostname together in either train or test.
hostnames = data["url"].map(
    lambda url: urlsplit(url).hostname.removeprefix("www.")
)

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
train_idx, test_idx = next(
    splitter.split(X, y, groups=hostnames)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(
    "\nHostnames shared between train and test:",
    len(set(hostnames.iloc[train_idx]) & set(hostnames.iloc[test_idx]))
)
print("\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))

print("\nTraining label counts:")
print(y_train.value_counts().sort_index())

print("\nTest label counts:")
print(y_test.value_counts().sort_index())

# Use part of the training set to compare model choices.
validation_groups = hostnames.iloc[train_idx]
validation_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=7
)
fit_idx, val_idx = next(
    validation_splitter.split(
        X_train,
        y_train,
        groups=validation_groups
    )
)

X_fit, y_fit = X_train.iloc[fit_idx], y_train.iloc[fit_idx]
X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

print("\nValidation label counts:")
print(y_val.value_counts().sort_index())

# Compare Logistic Regression with and without class weighting.
for weight in (None, "balanced"):
    candidate = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight=weight
        )
    )
    candidate.fit(X_fit, y_fit)
    val_pred = candidate.predict(X_val)

    print(f"\nLogistic Regression validation — class_weight={weight}:")
    print(classification_report(
        y_val,
        val_pred,
        labels=[0, 1],
        target_names=["Phishing (0)", "Legitimate (1)"],
        digits=3
    ))
    print(
        "False alarms:",
        ((y_val == 1) & (val_pred == 0)).sum()
    )
# Compare Decision Trees using all six URL features.
for tree_weight in (None, "balanced"):
    tree_candidate = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=100,
        class_weight=tree_weight,
        random_state=42
    )
    tree_candidate.fit(X_fit, y_fit)
    tree_pred = tree_candidate.predict(X_val)

    print(f"\nDecision Tree validation — class_weight={tree_weight}:")
    print(classification_report(
        y_val,
        tree_pred,
        labels=[0, 1],
        target_names=["Phishing (0)", "Legitimate (1)"],
        digits=3
    ))
    print(
        "False alarms:",
        ((y_val == 1) & (tree_pred == 0)).sum()
    )
# Compare phishing cutoffs on validation data only.
threshold_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=100,
    class_weight=None,
    random_state=42
)
threshold_model.fit(X_fit, y_fit)
val_phishing_scores = threshold_model.predict_proba(X_val)[:, 0]

print("\nCutoff | Phishing precision | Phishing recall | Phishing F1 | False alarms")
for cutoff in (0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10):
    cutoff_pred = pd.Series(
        (val_phishing_scores < cutoff).astype(int),
        index=y_val.index
    )
    result = classification_report(
        y_val,
        cutoff_pred,
        labels=[0, 1],
        target_names=["Phishing", "Benign"],
        output_dict=True,
        zero_division=0
    )
    phishing = result["Phishing"]
    false_alarms = ((y_val == 1) & (cutoff_pred == 0)).sum()

    print(
        f"{cutoff:6.2f} | {phishing['precision']:18.3f} | "
        f"{phishing['recall']:15.3f} | {phishing['f1-score']:11.3f} | "
        f"{false_alarms}"
    )

# Train the chosen model on all training rows.
model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=100,
    class_weight=None,
    random_state=42
)

model.fit(X_train, y_train)
print("\nTraining complete.")

# Evaluate it on hostnames held out from training.
# The 0.30 cutoff was chosen using validation results.
y_pred = (model.predict_proba(X_test)[:, 0] < 0.30).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print("\nTest accuracy (%):", round(accuracy * 100, 2))

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion matrix:")
print("Rows: actual [phishing (0), legitimate (1)]")
print("Columns: predicted [phishing (0), legitimate (1)]")
print(matrix)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Phishing (0)", "Legitimate (1)"],
    digits=3
))

print(
    "\nAlways-benign baseline accuracy (%):",
    round((y_test == 1).mean() * 100, 2)
)

# Column 0 contains the model's score for phishing (label 0).
phishing_probabilities = model.predict_proba(X_test)[:, 0]
actual_phishing = (y_test == 0).astype(int)

auc = roc_auc_score(
    actual_phishing,
    phishing_probabilities
)
print("\nROC-AUC for phishing:", round(auc, 3))


# Save the confusion matrix figure.
Path("plots").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Phishing (0)", "Legitimate (1)"]
)
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)
ax.set_title("Decision Tree Confusion Matrix")
fig.tight_layout()
fig.savefig(
    "plots/decision_tree_confusion_matrix.png",
    dpi=150
)
plt.close(fig)


# Save the ROC curve figure.
roc_fig, roc_ax = plt.subplots(figsize=(6, 5))

RocCurveDisplay.from_predictions(
    actual_phishing,
    phishing_probabilities,
    name="Decision Tree",
    ax=roc_ax
)

roc_ax.plot(
    [0, 1],
    [0, 1],
    "--",
    color="gray",
    label="Random guessing"
)
roc_ax.set_title("ROC Curve for Phishing Detection")
roc_ax.legend(loc="lower right")

roc_fig.tight_layout()
roc_fig.savefig(
    "plots/decision_tree_roc_curve.png",
    dpi=150
)
plt.close(roc_fig)

# Save the trained Decision Tree.
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/decision_tree_model.joblib")
print("\nSaved model: models/decision_tree_model.joblib")