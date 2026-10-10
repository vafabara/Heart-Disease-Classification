import math
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import Classification

# ----- Comparison tab constants (unchanged from the original app) -----
CHART_WIDTH = 420
BAR_MAX = 230
BAR_HEIGHT = 26
ROW_GAP = 46
BAR_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

CHOICES = [
    ("1. K-Nearest Neighbors (KNN)", "KNN"),
    ("2. Logistic Regression", "Logistic Regression"),
    ("3. Decision Tree", "Decision Tree"),
    ("4. Support Vector Machine (SVM)", "SVM"),
]

current_plot = {"canvas": None}

# ----- Prediction tab constants -----
YES_NO_OPTIONS = {"No (0)": 0, "Yes (1)": 1}
SEX_OPTIONS = {"Female (0)": 0, "Male (1)": 1}
CA_OPTIONS = {str(n): n for n in range(4)}
REST_ECG_OPTIONS = {
    "0 - Normal": 0,
    "1 - ST-T wave abnormality": 1,
    "2 - Left ventricular hypertrophy": 2,
}

# Editable example values the form starts with.
DEFAULT_VALUES = {
    "Age": 45, "Sex": 1, "exang": 0, "ca": 0, "cp": 2,
    "trtbps": 120, "chol": 200, "fbs": 0, "rest_ecg": 0, "thalach": 150,
}


def build_fields():
    # (key, label, kind, options). kind is "int", "float" or "choice".
    # Built at runtime because the chest pain codes come from the dataset.
    return [
        ("Age", "Age (years)", "int", None),
        ("Sex", "Sex", "choice", SEX_OPTIONS),
        ("exang", "Exercise-induced angina", "choice", YES_NO_OPTIONS),
        ("ca", "Major vessels (ca)", "choice", CA_OPTIONS),
        ("cp", "Chest pain type (cp)", "choice", Classification.cp_options()),
        ("trtbps", "Resting blood pressure (mm Hg)", "float", None),
        ("chol", "Serum cholesterol (mg/dl)", "float", None),
        ("fbs", "Fasting blood sugar > 120 mg/dl", "choice", YES_NO_OPTIONS),
        ("rest_ecg", "Resting ECG result", "choice", REST_ECG_OPTIONS),
        ("thalach", "Max heart rate achieved (bpm)", "float", None),
    ]


def label_for(options, value):
    for text, code in options.items():
        if code == value:
            return text
    return ""


# ===================== Prediction tab =====================
def read_patient(fields, variables):
    # Turns the form text into a clean dict, or raises ValueError with a readable message.
    patient = {}
    for key, label, kind, options in fields:
        text = variables[key].get().strip()
        if text == "":
            raise ValueError(f"'{label}' is empty.")

        if kind == "choice":
            if text not in options:
                raise ValueError(f"Please choose a valid option for '{label}'.")
            patient[key] = options[text]
        elif kind == "int":
            try:
                value = int(text)
            except ValueError:
                raise ValueError(f"'{label}' must be a whole number.") from None
            if not 1 <= value <= 120:
                raise ValueError(f"'{label}' must be between 1 and 120.")
            patient[key] = value
        else:
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"'{label}' must be a number.") from None
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"'{label}' must be a positive number.")
            patient[key] = value
    return patient


def format_prediction(result):
    lines = [
        f"Selected Model: {result['model']}",
        f"Predicted Target: {result['prediction']}",
        f"Result: {result['meaning']}",
        "",
        "Model-estimated class probabilities:",
    ]
    for target, probability in sorted(result["probabilities"].items()):
        lines.append(f"  Class {target} ({Classification.TARGET_MEANINGS[target]}): {probability * 100:.1f}%")
    lines += [
        "",
        "These are estimates from a simple ML model trained on a",
        "small dataset. This is not a medical diagnosis and does",
        "not predict whether a heart attack will occur.",
    ]
    return "\n".join(lines)


def build_prediction_tab(parent):
    tk.Label(parent, text="Heart Disease Prediction", font=("Arial", 16, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(0, 4))
    tk.Label(parent, text="Enter the patient's data, choose a model, then press Predict.",
             font=("Arial", 10), fg="gray").grid(row=1, column=0, columnspan=2, pady=(0, 12))

    fields = build_fields()
    variables = {}

    form = tk.Frame(parent)
    form.grid(row=2, column=0, sticky="n", padx=(0, 25))

    for row, (key, label, kind, options) in enumerate(fields):
        tk.Label(form, text=label, font=("Arial", 10), anchor="w").grid(row=row, column=0, sticky="w", pady=4)
        variable = tk.StringVar()
        if kind == "choice":
            variable.set(label_for(options, DEFAULT_VALUES[key]))
            widget = ttk.Combobox(form, textvariable=variable, values=list(options), state="readonly", width=30)
        else:
            variable.set(str(DEFAULT_VALUES[key]))
            widget = ttk.Entry(form, textvariable=variable, width=32)
        widget.grid(row=row, column=1, padx=(12, 0), pady=4)
        variables[key] = variable

    # Model selection sits below the 10 patient fields.
    model_row = len(fields)
    tk.Label(form, text="Model", font=("Arial", 10, "bold"), anchor="w").grid(
        row=model_row, column=0, sticky="w", pady=(14, 4))
    model_var = tk.StringVar(value="KNN")
    ttk.Combobox(form, textvariable=model_var, values=Classification.MODEL_NAMES,
                 state="readonly", width=30).grid(row=model_row, column=1, padx=(12, 0), pady=(14, 4))

    result_frame = tk.LabelFrame(parent, text="Prediction Result", font=("Arial", 10, "bold"), padx=12, pady=10)
    result_frame.grid(row=2, column=1, sticky="n")
    result_label = tk.Label(result_frame, text="No prediction yet.", font=("Courier", 10),
                            justify="left", anchor="nw", width=58, height=14)
    result_label.pack()

    def on_predict():
        try:
            patient = read_patient(fields, variables)
        except ValueError as error:
            result_label.config(text="No prediction: please fix the input.", fg="black")
            messagebox.showerror("Invalid input", str(error))
            return

        try:
            result = Classification.predict_heart_disease(patient, model_var.get())
        except Exception as error:
            result_label.config(text="Prediction failed.", fg="black")
            messagebox.showerror("Prediction failed", f"Could not run the model:\n{error}")
            return

        color = "#B22222" if result["prediction"] == 1 else "#2E7D32"
        result_label.config(text=format_prediction(result), fg=color)

    tk.Button(form, text="Predict", width=24, pady=6, font=("Arial", 10, "bold"),
              command=on_predict).grid(row=model_row + 1, column=0, columnspan=2, pady=(16, 0))


# ===================== Model comparison tab =====================
def draw_accuracy_chart(chart, benchmark):
    chart.create_text(CHART_WIDTH / 2, 18, text="Model Accuracy Comparison",
                      font=("Arial", 12, "bold"))
    for i, model_name in enumerate(Classification.MODEL_NAMES):
        accuracy = benchmark["models"][model_name]["metrics"]["Accuracy"]
        y = 50 + i * ROW_GAP
        chart.create_text(10, y + BAR_HEIGHT / 2, text=model_name, anchor="w", font=("Arial", 10))
        x0 = 150
        chart.create_rectangle(x0, y, x0 + accuracy * BAR_MAX, y + BAR_HEIGHT,
                               fill=BAR_COLORS[i], outline="")
        chart.create_text(x0 + accuracy * BAR_MAX + 8, y + BAR_HEIGHT / 2,
                          text=f"{accuracy * 100:.2f}%", anchor="w", font=("Arial", 10, "bold"))
    chart.create_text(CHART_WIDTH / 2, 50 + 4 * ROW_GAP - 6,
                      text=f"Bar scale: {BAR_MAX}px = 100% test accuracy",
                      font=("Arial", 8), fill="gray")


def show_model(benchmark, model_name, result_text, plot_frame):
    result_text.config(state="normal")
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, Classification.format_model_results(benchmark, model_name))
    result_text.config(state="disabled")

    if current_plot["canvas"] is not None:
        current_plot["canvas"].get_tk_widget().destroy()

    cm = benchmark["models"][model_name]["confusion_matrix"]
    fig = Classification.draw_confusion_matrix(cm, model_name)
    fig.set_size_inches(4.2, 3.6)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    plt.close(fig)
    current_plot["canvas"] = canvas


def build_comparison_tab(parent, benchmark, root):
    left = tk.Frame(parent, padx=15, pady=15)
    left.grid(row=0, column=0, sticky="n")
    right = tk.Frame(parent, padx=15, pady=15)
    right.grid(row=0, column=1, sticky="n")

    chart = tk.Canvas(left, width=CHART_WIDTH, height=50 + 4 * ROW_GAP, bg="white",
                      highlightthickness=1, highlightbackground="#cccccc")
    chart.pack()
    draw_accuracy_chart(chart, benchmark)

    tk.Label(left, text="MODEL COMPARISON", font=("Arial", 12, "bold")).pack(pady=(20, 8))

    result_text = tk.Text(right, width=42, height=14, font=("Courier", 10), state="disabled")
    plot_frame = tk.Frame(right)

    for label, model_name in CHOICES:
        tk.Button(left, text=label, width=34, pady=4,
                  command=lambda name=model_name: show_model(benchmark, name, result_text, plot_frame)
                  ).pack(pady=3)
    tk.Button(left, text="5. Exit", width=34, pady=4, command=root.destroy).pack(pady=3)

    result_text.pack()
    plot_frame.pack(pady=(10, 0))
    result_text.config(state="normal")
    result_text.insert(tk.END, "Choose a model on the left\nto see its results.")
    result_text.config(state="disabled")


def main():
    root = tk.Tk()
    root.withdraw()  # keep the window hidden while the models train
    root.title("Heart Disease Prediction")
    root.resizable(False, False)

    print("Training all models, please wait...")
    try:
        benchmark = Classification.run_benchmark()
    except Exception as error:
        messagebox.showerror("Startup error", f"Could not load the data and train the models:\n\n{error}")
        root.destroy()
        return

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    prediction_tab = tk.Frame(notebook, padx=20, pady=15)
    comparison_tab = tk.Frame(notebook)
    notebook.add(prediction_tab, text="New Patient Prediction")  # first tab = opens first
    notebook.add(comparison_tab, text="Model Comparison")

    build_prediction_tab(prediction_tab)
    build_comparison_tab(comparison_tab, benchmark, root)

    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()