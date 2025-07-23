import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Use OrdinalEncoder for font names (handles unseen fonts in future)
font_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df["font_name_encoded"] = font_encoder.fit_transform(df[["font_name"]])  # 2D input

# Use LabelEncoder for heading labels (H1, H2, BODY etc.)
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Features and target
X = df[["font_size", "font_name_encoded", "flags", "text_length", "spacing"]]
y = df["label_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save models
joblib.dump(clf, "heading_classifier.pkl")
joblib.dump(font_encoder, "font_encoder.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
