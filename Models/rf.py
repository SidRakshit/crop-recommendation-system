from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

data_file_name = '../Datasets/Modified_Crop_recommendation.csv'
data = pd.read_csv(data_file_name)
data = data.drop(columns=["N", "P", "K"])

X = data.drop(columns=["label"])
y = data["label"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

y_val_pred = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

print("\nRandom Forest Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

validation_results = X_val.copy()
validation_results['Actual'] = y_val.values
validation_results['Predicted'] = y_val_pred

validation_results['Correct'] = validation_results['Actual'] == validation_results['Predicted']

correct_predictions = validation_results[validation_results['Correct'] == True]
incorrect_predictions = validation_results[validation_results['Correct'] == False]

print(f"Correct Predictions: {len(correct_predictions)}")
print(f"Incorrect Predictions: {len(incorrect_predictions)}")

incorrect_predictions.to_csv("Incorrect_Predictions.csv", index=False)
print("Incorrect predictions saved to 'Incorrect_Predictions.csv'")