# Heart Disease Classification ❤️

A beginner-friendly Machine Learning project that predicts heart disease risk classes for new patients and compares four classification algorithms.

## 🖥️ GUI Preview

![Application GUI](assets/gui.png)

## 🩺 Patient Prediction

Enter a patient's health data, select a trained model, and predict the patient's risk class:

- `0` — Low risk
- `1` — High risk

## 📊 Model Comparison

<table>
  <tr>
    <td align="center"><b>KNN</b></td>
    <td align="center"><b>Logistic Regression</b></td>
    <td align="center"><b>Decision Tree</b></td>
    <td align="center"><b>SVM</b></td>
  </tr>
  <tr>
    <td><img src="assets/knn_confusion_matrix.png" width="180" alt="KNN Confusion Matrix"></td>
    <td><img src="assets/logistic_regression_confusion_matrix.png" width="180" alt="Logistic Regression Confusion Matrix"></td>
    <td><img src="assets/decision_tree_confusion_matrix.png" width="180" alt="Decision Tree Confusion Matrix"></td>
    <td><img src="assets/svm_confusion_matrix.png" width="180" alt="SVM Confusion Matrix"></td>
  </tr>
</table>

Models are evaluated using Accuracy, Precision, Recall, F1 Score, Log Loss, and Confusion Matrices.

## 🤖 Algorithms

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)

## ⚙️ Installation & Usage

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the application:

```bash
python main.py
```

## 🛠️ Technologies

Python · Pandas · NumPy · Scikit-learn · Matplotlib