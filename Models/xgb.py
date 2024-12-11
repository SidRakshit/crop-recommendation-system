from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data_file_name = '../Datasets/Modified_Crop_recommendation.csv'
data = pd.read_csv(data_file_name)
data = data.drop(columns=["N", "P", "K"])

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

X = data.drop(columns=["label"])
y = data["label"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_train_pred = xgb_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

y_val_pred = xgb_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

y_test_pred = xgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

print("\nXGBoost Test Set Classification Report:")
print(classification_report(y_test_labels, y_test_pred_labels, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test_labels, y_test_pred_labels, labels=label_encoder.classes_))

validation_results = X_val.copy()
validation_results['Actual'] = label_encoder.inverse_transform(y_val.values)
validation_results['Predicted'] = label_encoder.inverse_transform(y_val_pred)
validation_results['Correct'] = validation_results['Actual'] == validation_results['Predicted']

correct_predictions = validation_results[validation_results['Correct'] == True]
incorrect_predictions = validation_results[validation_results['Correct'] == False]

print(f"Correct Predictions: {len(correct_predictions)}")
print(f"Incorrect Predictions: {len(incorrect_predictions)}")

incorrect_predictions.to_csv("Incorrect_Predictions_XGBoost.csv", index=False)
print("Incorrect predictions saved to 'Incorrect_Predictions_XGBoost.csv'")
