import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import os


# Load dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# EDA
print('Shape:', data.shape)
print('\nData Types:\n', data.dtypes)
print('\nMissing Values:\n', data.isnull().sum())
print('\nDuplicates:', data.duplicated().sum())
print('\nDescribe:\n', data.describe())

# Prepare data
X = data.drop('label', axis=1)
y = data['label']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Base Learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf', C=1.0)))
]

# Build Stacked Ensemble Model
stacked_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=False,
    n_jobs=-1
)

# Train the Model
stacked_model.fit(X_train, y_train)

# Evaluate the model
y_pred = stacked_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print('\nClassification Report:\n', report)

# Feature Importance - Random Forest
rf_model = stacked_model.named_estimators_['rf']
importances = rf_model.feature_importances_
features = data.columns[:-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, hue=importances, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
os.makedirs('visuals', exist_ok=True)
plt.savefig('visuals/feature_importance.png')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, xticks_rotation=80, cmap='Blues', colorbar=True)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('visuals/confusion_matrix.png')
plt.close()

# Save the Model and Label Encoder
os.makedirs('models', exist_ok=True)
joblib.dump(stacked_model, 'models/smart_crop_recommender.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
print("\nModel saved to models/smart_crop_recommender.pkl")
print("Label encoder saved to models/label_encoder.pkl")
print("Feature importance plot saved to visuals/feature_importance.png")
print("Confusion matrix plot saved to visuals/confusion_matrix.png")