import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode categorical variables
le_font = LabelEncoder()
df["font_name_encoded"] = le_font.fit_transform(df["font_name"])

le_label = LabelEncoder()
df["label_encoded"] = le_label.fit_transform(df["label"])

# Features and target
X = df[["font_size", "font_name_encoded", "flags", "text_length", "spacing"]]
y = df["label_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le_label.classes_))

# Save model
joblib.dump(clf, "heading_classifier.pkl")
joblib.dump(le_font, "font_encoder.pkl")
joblib.dump(le_label, "label_encoder.pkl")
