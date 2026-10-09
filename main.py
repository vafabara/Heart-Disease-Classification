import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import Classification

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


def main():
    print("Training and evaluating all models, please wait...")
    benchmark = Classification.run_benchmark()

    root = tk.Tk()
    root.title("Heart Disease Classification")
    root.resizable(False, False)

    left = tk.Frame(root, padx=15, pady=15)
    left.grid(row=0, column=0, sticky="n")
    right = tk.Frame(root, padx=15, pady=15)
    right.grid(row=0, column=1, sticky="n")

    chart = tk.Canvas(left, width=CHART_WIDTH, height=50 + 4 * ROW_GAP, bg="white",
                      highlightthickness=1, highlightbackground="#cccccc")
    chart.pack()
    draw_accuracy_chart(chart, benchmark)

    tk.Label(left, text="HEART DISEASE CLASSIFICATION", font=("Arial", 12, "bold")).pack(pady=(20, 8))

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

    root.mainloop()


if __name__ == "__main__":
    main()
