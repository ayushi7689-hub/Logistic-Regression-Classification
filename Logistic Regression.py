import pandas as py
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer 
from sklearn.metrics import classification_report

dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

print("Dataset Shape:", X.shape)
print("\nFirst 5 Rows:")
print(X.head())

print("\nTarget Classes:", dataset.target_names)

from sklearn.model_selection import train_test_split

#splitting data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify= y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


from sklearn.preprocessing import StandardScaler

# Create a standardization tool
scaler = StandardScaler()

# Learn the scale from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scale to testing data
X_test_scaled = scaler.transform(X_test)

print("Standardization completed!")
print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)


from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Teach the model using training data
model.fit(X_train_scaled, y_train)

print("Logistic Regression model trained successfully!")

y_pred = model.predict(X_test_scaled)

print("First 10 Prediction:", y_pred[:10])
print("First 10 Actual Answers:", y_test.iloc[:10].values)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Persentage:", accuracy * 100, "%")


from sklearn.metrics import precision_score, recall_score

# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=dataset.target_names
))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Display the matrix as a graph
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.target_names
)

display.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve

# Get predicted probabilities for class 1 (benign)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC-AUC score
auc_score = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", auc_score)

# Draw the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


from sklearn.metrics import precision_score, recall_score

# Choose a new threshold
threshold = 0.4

# Make predictions using the new threshold
y_pred_tuned = (y_prob >= threshold).astype(int)

print("Threshold:", threshold)

print("Precision:", precision_score(y_test, y_pred_tuned))
print("Recall:", recall_score(y_test, y_pred_tuned))

print("\nConfusion Matrix with New Threshold:")
print(confusion_matrix(y_test, y_pred_tuned))

