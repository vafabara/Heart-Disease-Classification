# ===== Stage 1: Import =====
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart.csv"
PLOTS_DIR = BASE_DIR / "plots"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
K_VALUES = list(range(1, 26))
TREE_DEPTHS = list(range(2, 9))

MODEL_NAMES = ["KNN", "Logistic Regression", "Decision Tree", "SVM"]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1 Score", "Log Loss"]

# The 10 predictor features, in the exact order the models are trained on.
FEATURE_COLUMNS = ["Age", "Sex", "exang", "ca", "cp", "trtbps", "chol", "fbs", "rest_ecg", "thalach"]

TARGET_MEANINGS = {0: "Low-risk class", 1: "High-risk class"}

# Chest pain descriptions in dataset order (first code = typical angina, last = asymptomatic).
CP_DESCRIPTIONS = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]

# The CSV uses slightly different column names than the project description.
COLUMN_RENAMES = {
    "age": "Age",
    "sex": "Sex",
    "exng": "exang",
    "caa": "ca",
    "restecg": "rest_ecg",
    "thalachh": "thalach",
    "output": "target",
}

# Filled in by run_benchmark(). Holds the trained models so they are trained only once.
_benchmark = None


# ===== Stage 2: Load Data =====
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}\nPut heart.csv in the same folder as main.py.")
    df = pd.read_csv(DATA_PATH).rename(columns=COLUMN_RENAMES)
    missing = [column for column in FEATURE_COLUMNS + ["target"] if column not in df.columns]
    if missing:
        raise ValueError(f"heart.csv is missing expected column(s): {missing}\nColumns found: {list(df.columns)}")
    return df


def inspect_data(df):
    target_counts = df["target"].value_counts().sort_index()
    print("========== DATASET INSPECTION ==========")
    print(f"Rows x columns       : {df.shape[0]} x {df.shape[1]}")
    print(f"Missing values       : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows       : {int(df.duplicated().sum())} (kept, not removed)")
    print(f"Non-numeric columns  : {list(df.select_dtypes(exclude='number').columns) or 'none'}")
    print(f"Target = 0 / Target = 1: {target_counts[0]} / {target_counts[1]} "
          f"({target_counts[1] / len(df):.1%} positive)")
    print("=========================================\n")


def detect_cp_offset(df):
    # The first chest-pain code is either 0 or 1 depending on the dataset version.
    # Reading it from the data keeps the labels (typical angina ... asymptomatic) correct
    # without silently shifting what each code means.
    low, high = int(df["cp"].min()), int(df["cp"].max())
    if low not in (0, 1) or high != low + 3:
        raise ValueError(f"Unexpected 'cp' values in heart.csv: found {low} to {high}, expected 0-3 or 1-4.")
    return low


# ===== Stage 3: Prepare X and y =====
def prepare_x_y(df):
    # Only the 10 chosen predictors go into X. 'target' and any extra columns stay out.
    X = df[FEATURE_COLUMNS]
    y = df["target"]
    return X, y


# ===== Stage 4: Train/Test Split =====
def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# ===== Stage 5: Scaling =====
# Scaling now lives inside each model's Pipeline (see the train_* functions below).
# A Pipeline fits the scaler on the training data only, then reuses those same
# learned values for every later prediction, including a single new patient.


# ===== Stage 6: KNN =====
def find_best_k(X_train, y_train):
    # Scaler is re-fitted inside every fold so validation rows never influence scaling.
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fit_idx, val_idx in skf.split(X_train, y_train):
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_train.iloc[fit_idx])
        X_val = scaler.transform(X_train.iloc[val_idx])
        folds.append((X_fit, y_train.iloc[fit_idx], X_val, y_train.iloc[val_idx]))

    mean_f1_scores = []
    for k in K_VALUES:
        fold_scores = []
        for X_fit, y_fit, X_val, y_val in folds:
            knn = KNeighborsClassifier(n_neighbors=k).fit(X_fit, y_fit)
            fold_scores.append(f1_score(y_val, knn.predict(X_val), zero_division=0))
        mean_f1_scores.append(float(np.mean(fold_scores)))

    best_k = K_VALUES[int(np.argmax(mean_f1_scores))]
    return best_k, mean_f1_scores


def train_knn(X_train, y_train, best_k):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_train, y_train)


# ===== Stage 7: Logistic Regression =====
def train_logistic_regression(X_train, y_train):
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    ).fit(X_train, y_train)


# ===== Stage 8: Decision Tree =====
def find_best_tree_depth(X_train, y_train):
    # An unrestricted tree memorises the training data and outputs only 0/1 probabilities,
    # which makes Log Loss meaningless, so depth is chosen on training data only.
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mean_scores = []
    for depth in TREE_DEPTHS:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        scores = cross_val_score(tree, X_train, y_train, cv=skf, scoring="f1")
        mean_scores.append(scores.mean())
    return TREE_DEPTHS[int(np.argmax(mean_scores))]


def train_decision_tree(X_train, y_train, max_depth):
    # Trees split on thresholds, so scaling would change nothing: no scaler here.
    return make_pipeline(DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)).fit(X_train, y_train)


# ===== Stage 9: SVM =====
def train_svm(X_train, y_train):
    # probability=True is required so SVC can provide probabilities for Log Loss.
    return make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    ).fit(X_train, y_train)


# ===== Stage 10: Evaluation =====
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Log Loss": log_loss(y_test, y_proba, labels=[0, 1]),
    }
    return {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }


# ===== Stage 11: Comparison =====
def print_comparison_table(benchmark):
    print("========== MODEL COMPARISON (test set, positive class: target = 1) ==========")
    print(f"{'Model':<22}" + "".join(f"{name:>12}" for name in METRIC_NAMES))
    for model_name in MODEL_NAMES:
        metrics = benchmark["models"][model_name]["metrics"]
        print(f"{model_name:<22}" + "".join(f"{metrics[name]:>12.4f}" for name in METRIC_NAMES))
    print()


# ===== Plots =====
def draw_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (lower risk)", "1 (higher risk)"])
    ax.set_yticklabels(["0 (lower risk)", "1 (higher risk)"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{model_name} - Confusion Matrix")
    threshold = cm.max() / 2
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=16,
                    color="white" if cm[row, col] > threshold else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def draw_metric_bars(benchmark, metric_name, filename_title):
    values = [benchmark["models"][name]["metrics"][metric_name] for name in MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(MODEL_NAMES, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                ha="center", va="bottom")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    ax.set_title(filename_title)
    plt.tight_layout()
    return fig


def draw_k_vs_f1(benchmark):
    best_k = benchmark["best_k"]
    scores = benchmark["k_scores"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_VALUES, scores, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
    ax.set_xlabel("K (n_neighbors)")
    ax.set_ylabel("Mean F1 Score (cross-validation on training data)")
    ax.set_title("KNN - K vs F1 Score")
    ax.set_xticks(K_VALUES[::2])
    ax.legend()
    plt.tight_layout()
    return fig


def save_figure(fig, filename):
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def save_all_plots(benchmark):
    PLOTS_DIR.mkdir(exist_ok=True)
    file_names = {
        "KNN": "knn_confusion_matrix.png",
        "Logistic Regression": "logistic_regression_confusion_matrix.png",
        "Decision Tree": "decision_tree_confusion_matrix.png",
        "SVM": "svm_confusion_matrix.png",
    }
    for model_name, filename in file_names.items():
        cm = benchmark["models"][model_name]["confusion_matrix"]
        save_figure(draw_confusion_matrix(cm, model_name), filename)
    save_figure(draw_metric_bars(benchmark, "F1 Score", "Model Comparison - F1 Score"), "f1_comparison.png")
    save_figure(draw_metric_bars(benchmark, "Accuracy", "Model Comparison - Test Accuracy"), "accuracy_comparison.png")
    save_figure(draw_k_vs_f1(benchmark), "knn_k_vs_f1.png")


# ===== Training + evaluation (runs once at startup) =====
def run_benchmark(verbose=True):
    global _benchmark

    df = load_data()
    if verbose:
        inspect_data(df)
    cp_offset = detect_cp_offset(df)

    X, y = prepare_x_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_k, k_scores = find_best_k(X_train, y_train)
    best_depth = find_best_tree_depth(X_train, y_train)

    knn = train_knn(X_train, y_train, best_k)
    logistic = train_logistic_regression(X_train, y_train)
    tree = train_decision_tree(X_train, y_train, best_depth)
    svm = train_svm(X_train, y_train)

    # Every model is a Pipeline, so it accepts raw (unscaled) rows and scales internally.
    benchmark = {
        "best_k": best_k,
        "k_scores": k_scores,
        "tree_depth": best_depth,
        "cp_offset": cp_offset,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "models": {
            "KNN": evaluate_model(knn, X_test, y_test),
            "Logistic Regression": evaluate_model(logistic, X_test, y_test),
            "Decision Tree": evaluate_model(tree, X_test, y_test),
            "SVM": evaluate_model(svm, X_test, y_test),
        },
    }

    try:
        save_all_plots(benchmark)
    except OSError as error:
        # Saving PNGs is a bonus; it must not stop the app from starting.
        print(f"Warning: could not save plots ({error})")

    _benchmark = benchmark
    return benchmark


# ===== Functions called by main.py =====
def get_benchmark():
    if _benchmark is None:
        raise RuntimeError("Models are not trained yet. Call run_benchmark() first.")
    return _benchmark


def cp_options():
    # {"1 - Typical angina": 1, ...} using the codes that actually exist in heart.csv.
    offset = get_benchmark()["cp_offset"]
    return {f"{offset + i} - {text}": offset + i for i, text in enumerate(CP_DESCRIPTIONS)}


def predict_heart_disease(patient_data, model_name):
    benchmark = get_benchmark()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name!r}. Choose one of {MODEL_NAMES}.")
    missing = [name for name in FEATURE_COLUMNS if name not in patient_data]
    if missing:
        raise ValueError(f"Missing patient value(s): {missing}")
    if patient_data["cp"] not in cp_options().values():
        raise ValueError(f"Invalid chest pain code: {patient_data['cp']}")

    # One-row DataFrame with the same columns, in the same order, as the training data.
    row = pd.DataFrame([[patient_data[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    model = benchmark["models"][model_name]["model"]
    predicted = int(model.predict(row)[0])
    probabilities = {int(c): float(p) for c, p in zip(model.classes_, model.predict_proba(row)[0])}
    return {
        "model": model_name,
        "prediction": predicted,
        "meaning": TARGET_MEANINGS[predicted],
        "probabilities": probabilities,
    }


def format_model_results(benchmark, model_name):
    result = benchmark["models"][model_name]
    cm = result["confusion_matrix"]

    lines = [f"{model_name} (positive class: target = 1)", ""]
    if model_name == "KNN":
        lines.append(f"Selected K: {benchmark['best_k']}")
    if model_name == "Decision Tree":
        lines.append(f"Selected max_depth: {benchmark['tree_depth']}")
    for name in METRIC_NAMES:
        lines.append(f"{name:<10}: {result['metrics'][name]:.4f}")
    lines += [
        "",
        "Confusion Matrix (rows = actual):",
        "            Pred 0  Pred 1",
        f"Actual 0    {cm[0, 0]:>6}  {cm[0, 1]:>6}",
        f"Actual 1    {cm[1, 0]:>6}  {cm[1, 1]:>6}",
    ]
    return "\n".join(lines)