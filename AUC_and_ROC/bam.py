from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Example true labels and predicted probabilities
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# AUC Score
auc = roc_auc_score(y_true, y_scores)
print("AUC Score:", auc)

# Plot ROC
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')  # random guessing line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
