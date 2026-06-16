import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# -------------------------------------------------
# Example Data (Replace with your model outputs)
# -------------------------------------------------

# Ground truth labels
# 0 = Real
# 1 = Fake
y_true = [0,1,0,1,0,1,1,0,1,0]

# Model predictions
y_pred = [0,1,0,0,0,1,1,0,1,1]

# Prediction probabilities
y_scores = [0.1,0.9,0.2,0.4,0.3,0.8,0.9,0.2,0.7,0.6]


# -------------------------------------------------
# Create Results Folder
# -------------------------------------------------

output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)


# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real","Fake"],
    yticklabels=["Real","Fake"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))
plt.close()


# -------------------------------------------------
# Classification Report
# -------------------------------------------------

report = classification_report(y_true, y_pred)

with open(os.path.join(output_dir,"classification_report.txt"),"w") as f:
    f.write(report)


# -------------------------------------------------
# Precision Recall F1 Score
# -------------------------------------------------

precision = precision_score(y_true,y_pred)
recall = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)
accuracy = accuracy_score(y_true,y_pred)

metrics = {
    "Precision":precision,
    "Recall":recall,
    "F1 Score":f1,
    "Accuracy":accuracy
}

plt.figure(figsize=(7,5))

plt.bar(metrics.keys(),metrics.values())

plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Evaluation Metrics")

plt.savefig(os.path.join(output_dir,"evaluation_metrics.png"))
plt.close()


# -------------------------------------------------
# ROC Curve
# -------------------------------------------------

fpr,tpr,thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr,tpr)

plt.figure()

plt.plot(fpr,tpr,label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")
plt.legend()

plt.savefig(os.path.join(output_dir,"roc_curve.png"))
plt.close()


# -------------------------------------------------
# Print Results
# -------------------------------------------------

print("\nEvaluation Results")
print("---------------------------")
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",f1)
print("Accuracy:",accuracy)

print("\nAll graphs saved in folder:",output_dir)