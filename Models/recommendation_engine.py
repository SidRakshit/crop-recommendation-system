import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

data_file_name = '../Datasets/Modified_Crop_recommendation.csv'
data = pd.read_csv(data_file_name)
data = data.drop(columns=["N", "P", "K"])

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

X = data.drop(columns=["label"])
y = data["label"]

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X, y)

def recommend_crop(model, input_features, label_encoder):
    input_df = pd.DataFrame([input_features], columns=X.columns)
    prediction = model.predict(input_df)[0]
    recommended_crop = label_encoder.inverse_transform([prediction])[0]
    return recommended_crop

print("Enter the following details for crop recommendation:")

user_input = {
    "temperature": float(input("Temperature (°C): ")),
    "humidity": float(input("Humidity (%): ")),
    "ph": float(input("Soil pH: ")),
    "rainfall": float(input("Rainfall (mm): ")),
    "NPK_sum": float(input("Total NPK value: ")),
    "NPK_mean": float(input("Average NPK value: ")),
    "NPK_weighted": float(input("Weighted NPK value: ")),
    "NPK_weighted_average": float(input("Weighted Average NPK value: "))
}

recommended_crop = recommend_crop(xgb_model, user_input, label_encoder)
print(f"\nRecommended Crop: {recommended_crop}")
