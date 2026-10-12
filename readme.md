# Heart Disease Classification ❤️

A beginner-friendly Machine Learning project that predicts heart disease risk classes and compares four classification algorithms.

## 🖥️ GUI Preview

![Application GUI](assets/gui.png)

## 🩺 Patient Prediction & Dataset Management

- Predict patient risk: `0` (Low Risk) or `1` (High Risk).
- Add new patients directly through the GUI and save records to `heart.csv`.
- Select the verified class label (`0` or `1`) for each new patient.
- Automatically retrain all four models and update performance comparisons.

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

## ⚙️ Installation & Usage

```bash
pip install pandas numpy matplotlib scikit-learn
python main.py
```

## 🛠️ Technologies

Python · Pandas · NumPy · Scikit-learn · Matplotlib