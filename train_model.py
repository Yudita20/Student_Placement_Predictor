import pandas as pd
import matplotlib.pyplot as plt

# Loading Dataset
df = pd.read_csv("Placement_Data.csv")
print(df.columns)
print(df.head())

# Show dataset info
print(df.info())

# Describe
print(df.describe())

# Status Visualization
(df["status"].value_counts().reindex(["Placed", "Not Placed"]).plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["green", "yellow"]
))
plt.title("Student placement status")
plt.show()

# Data Cleaning and preprocessing

# Remove unnecessary columns
df.drop("sl_no", axis=1, inplace=True)

# Salary is available only for placed students, so remove it
df.drop("salary", axis=1, inplace=True)

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Label Encoding
le = LabelEncoder()

categorical_cols = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
    "status"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Numerical columns
num_cols = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p"
]

# Separate features and target
X = df.drop("status", axis=1)
Y = df["status"]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Scale numerical columns
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)

# Prediction
Y_pdt = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(Y_test, Y_pdt)
print("Accuracy:", accuracy)

cm = confusion_matrix(Y_test, Y_pdt)
print(cm)

print(classification_report(Y_test, Y_pdt))

# Save Model and Scaler
import joblib

joblib.dump(model, "placement_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load Model and Scaler
loaded_model = joblib.load("placement_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# New Student Data
new_student = pd.DataFrame({
    "gender": [1],
    "ssc_p": [85],
    "ssc_b": [1],
    "hsc_p": [80],
    "hsc_b": [1],
    "hsc_s": [2],
    "degree_p": [78],
    "degree_t": [2],
    "workex": [1],
    "etest_p": [75],
    "specialisation": [0],
    "mba_p": [70]
})

# Scale numerical columns
new_student[num_cols] = loaded_scaler.transform(new_student[num_cols])

# Predict
prediction = loaded_model.predict(new_student)

if prediction[0] == 1:
    print("Placed")
else:
    print("Not Placed")