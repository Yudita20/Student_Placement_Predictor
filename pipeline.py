import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

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


cat_cols = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation"
]

# Numerical columns
num_cols = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p"
]

preprocessor = ColumnTransformer(
    transformers =[
        ("num" , StandardScaler() , num_cols),
        ("cat" , OneHotEncoder(handle_unknown = "ignore") , cat_cols)
    ]
)

# Separate features and target
X = df.drop("status", axis=1)
Y = df["status"]

target_encoder = LabelEncoder()
Y = target_encoder.fit_transform(Y)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# Train Model
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("preprocessor" , preprocessor),
    ("model" , LogisticRegression())
])


pipeline.fit(X_train, Y_train)

# Prediction
Y_pdt = pipeline.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(Y_test, Y_pdt)
print("Accuracy:", accuracy)

cm = confusion_matrix(Y_test, Y_pdt)
print(cm)

print(classification_report(Y_test, Y_pdt))

joblib.dump(pipeline , "placement_pipeline.pkl")
joblib.dump(target_encoder , "target_encoder.pkl")

# Load Pipeline and Target Encoder
loaded_pipeline = joblib.load("placement_pipeline.pkl")
loaded_target_encoder = joblib.load("target_encoder.pkl")

new_student = pd.DataFrame({
    "gender": ["M"],
    "ssc_p": [85],
    "ssc_b": ["Central"],
    "hsc_p": [80],
    "hsc_b": ["Central"],
    "hsc_s": ["Science"],
    "degree_p": [78],
    "degree_t": ["Sci&Tech"],
    "workex": ["Yes"],
    "etest_p": [75],
    "specialisation": ["Mkt&Fin"],
    "mba_p": [70]
})

prediction = loaded_pipeline.predict(new_student)
result = loaded_target_encoder.inverse_transform(prediction)
print(result[0])

print(df["gender"].unique())
print(df["ssc_b"].unique())
print(df["hsc_b"].unique())
print(df["hsc_s"].unique())
print(df["degree_t"].unique())
print(df["workex"].unique())
print(df["specialisation"].unique())

