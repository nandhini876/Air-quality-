# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Upload the CSV file
uploaded = files.upload()

# Load the dataset (after uploading)
filename = list(uploaded.keys())[0]  # Automatically pick the uploaded file name
data = pd.read_csv(filename)

# Data Preprocessing
# Encoding categorical features
label_encoder_country = LabelEncoder()
label_encoder_status = LabelEncoder()
data['Country'] = label_encoder_country.fit_transform(data['Country'])
data['Status'] = label_encoder_status.fit_transform(data['Status'])

# Extract date features
date_features = pd.to_datetime(data['Date'])
data['Year'] = date_features.dt.year
data['Month'] = date_features.dt.month
data['Day'] = date_features.dt.day
data.drop(columns=['Date'], inplace=True)

# Splitting the data into features (X) and target (y)
X = data.drop(columns=['Status'])
y = data['Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
