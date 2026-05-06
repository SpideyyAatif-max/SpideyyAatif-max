import numpy as np

# States: 0=Sunny, 1=Cloudy, 2=Rainy
states = ["Sunny", "Cloudy", "Rainy"]

# Transition Matrix
T = np.array([
    [0.6, 0.3, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])

def simulate(days=10):
    current = 0  # start with Sunny
    rainy_days = 0

    weather_seq = []

    for _ in range(days):
        weather_seq.append(states[current])

        if current == 2:
            rainy_days += 1

        current = np.random.choice([0,1,2], p=T[current])

    return rainy_days, weather_seq


# Run multiple simulations to estimate probability
trials = 10000
count = 0

for _ in range(trials):
    rainy_days, _ = simulate()

    if rainy_days >= 3:
        count += 1

probability = count / trials

print("Estimated Probability (≥ 3 rainy days):", probability)

# Example one simulation output
rainy_days, seq = simulate()
print("\nSample 10-day Weather:")
print(seq)
print("Rainy Days:", rainy_days)

-----------------------------------------------------------------------------------------------------------

from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Structure
model = BayesianModel([
    ('Disease', 'Fever'),
    ('Disease', 'Cough'),
    ('Disease', 'Fatigue'),
    ('Disease', 'Chills')
])

# Disease prior
cpd_Disease = TabularCPD('Disease', 2, [[0.3], [0.7]])  # Flu, Cold

# Fever
cpd_Fever = TabularCPD(
    'Fever', 2,
    [[0.9, 0.5],   # Yes
     [0.1, 0.5]],  # No
    evidence=['Disease'],
    evidence_card=[2]
)

# Cough
cpd_Cough = TabularCPD(
    'Cough', 2,
    [[0.8, 0.6],
     [0.2, 0.4]],
    evidence=['Disease'],
    evidence_card=[2]
)

# Fatigue
cpd_Fatigue = TabularCPD(
    'Fatigue', 2,
    [[0.7, 0.3],
     [0.3, 0.7]],
    evidence=['Disease'],
    evidence_card=[2]
)

# Chills
cpd_Chills = TabularCPD(
    'Chills', 2,
    [[0.6, 0.4],
     [0.4, 0.6]],
    evidence=['Disease'],
    evidence_card=[2]
)

# Add CPDs
model.add_cpds(cpd_Disease, cpd_Fever, cpd_Cough, cpd_Fatigue, cpd_Chills)

# Inference
infer = VariableElimination(model)

# Task: P(Disease | Fever=Yes, Cough=Yes)
result = infer.query(
    variables=['Disease'],
    evidence={'Fever':0, 'Cough':0}
)

print("P(Disease | Fever=Yes, Cough=Yes):")
print(result)

---------------------------------------------------------------------------------------------------------

# Employee Attrition Dataset - Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

# ==============================
# 1. Dataset Overview
# ==============================

# Load dataset
df = pd.read_csv("employee_attrition.csv")   # change filename if needed

print("Dataset Shape:")
print(df.shape)

print("\nTotal Records:", df.shape[0])
print("Total Attributes:", df.shape[1])

print("\nColumn Data Types:")
print(df.dtypes)

print("\nFirst Five Rows:")
print(df.head())

print("\nBasic Dataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ==============================
# 2. Missing & Inconsistent Data
# ==============================

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percent
})

missing_table = missing_table[missing_table["Missing Count"] > 0]
print("\nMissing Values Summary:")
print(missing_table.sort_values(by="Missing Percentage", ascending=False))

# Check duplicate rows
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

# Drop duplicates if any
df = df.drop_duplicates()

# Check unique values in categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns

print("\nUnique Values in Categorical Columns:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())


# ==============================
# 3. Univariate Analysis
# ==============================

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

print("\nNumerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# Skewness of numerical columns
print("\nSkewness of Numerical Attributes:")
print(df[numerical_cols].skew().sort_values(ascending=False))

# Histograms for numerical columns
for col in numerical_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Count plots for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Count Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


# Specific categorical imbalance analysis
important_cats = ["Department", "Gender", "JobRole", "WorkLifeBalance"]

for col in important_cats:
    if col in df.columns:
        print(f"\nValue Counts for {col}:")
        print(df[col].value_counts(normalize=True) * 100)


# ==============================
# 4. Bivariate & Multivariate Analysis
# ==============================

# Convert Attrition to numeric if needed
if df["Attrition"].dtype == "object":
    df["Attrition"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0
    })

print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())

# Attrition count plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Attrition")
plt.title("Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()

# Relationship: OverTime vs Attrition
if "OverTime" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="OverTime", hue="Attrition")
    plt.title("OverTime vs Attrition")
    plt.xlabel("OverTime")
    plt.ylabel("Count")
    plt.show()

# Relationship: JobSatisfaction vs Attrition
if "JobSatisfaction" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="JobSatisfaction", hue="Attrition")
    plt.title("Job Satisfaction vs Attrition")
    plt.xlabel("Job Satisfaction")
    plt.ylabel("Count")
    plt.show()

# Relationship: MonthlyIncome vs Attrition
if "MonthlyIncome" in df.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
    plt.title("Monthly Income vs Attrition")
    plt.xlabel("Attrition")
    plt.ylabel("Monthly Income")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 10))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# ==============================
# 5. Group-Based Comparison
# ==============================

comparison_cols = ["Age", "YearsAtCompany", "MonthlyIncome", "DistanceFromHome"]

available_cols = [col for col in comparison_cols if col in df.columns]

group_comparison = df.groupby("Attrition")[available_cols].mean()

print("\nGroup-Based Mean Comparison:")
print(group_comparison)

for col in available_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Attrition", y=col)
    plt.title(f"{col} Comparison by Attrition")
    plt.xlabel("Attrition")
    plt.ylabel(col)
    plt.show()


# ==============================
# 6. Outlier Detection
# ==============================

outlier_cols = [
    "MonthlyIncome",
    "DistanceFromHome",
    "NumCompaniesWorked",
    "YearsAtCompany"
]

outlier_cols = [col for col in outlier_cols if col in df.columns]

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound

for col in outlier_cols:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    
    print(f"\nOutlier Analysis for {col}:")
    print("Lower Bound:", lower)
    print("Upper Bound:", upper)
    print("Number of Outliers:", len(outliers))
    print("Outlier Percentage:", round((len(outliers) / len(df)) * 100, 2), "%")
    
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, y=col)
    plt.title(f"Boxplot for Outlier Detection: {col}")
    plt.ylabel(col)
    plt.show()


# Z-score method
from scipy.stats import zscore

print("\nZ-Score Based Outlier Detection:")

for col in outlier_cols:
    z_scores = np.abs(zscore(df[col].dropna()))
    outlier_count = np.sum(z_scores > 3)
    
    print(f"{col}: {outlier_count} outliers using Z-score method")


# ==============================
# 7. Required Visualizations
# ==============================

# Visualization 1: Attrition distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Attrition")
plt.title("Visualization 1: Attrition Distribution")
plt.show()

# Visualization 2: Age distribution
if "Age" in df.columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(df["Age"], kde=True)
    plt.title("Visualization 2: Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

# Visualization 3: MonthlyIncome by Attrition
if "MonthlyIncome" in df.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
    plt.title("Visualization 3: Monthly Income by Attrition")
    plt.show()

# Visualization 4: Department vs Attrition
if "Department" in df.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="Department", hue="Attrition")
    plt.title("Visualization 4: Department vs Attrition")
    plt.xticks(rotation=45)
    plt.show()

# Visualization 5: Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Visualization 5: Correlation Heatmap")
plt.show()

# Optional Visualization 6: Pairplot for selected variables
pairplot_cols = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome", "Attrition"]
pairplot_cols = [col for col in pairplot_cols if col in df.columns]

if len(pairplot_cols) >= 3:
    sns.pairplot(df[pairplot_cols], hue="Attrition")
    plt.show()


# ==============================
# 8. Final Key Observations Helper
# ==============================

print("\nFinal EDA Summary Points:")

print("""
1. The dataset was loaded successfully and basic structure was examined.
2. Missing values and duplicate records were checked.
3. Numerical variables were analyzed using histograms, skewness, and summary statistics.
4. Categorical variables were analyzed using count plots and percentage distributions.
5. Attrition was compared with important variables such as OverTime, JobSatisfaction, and MonthlyIncome.
6. Group-based comparison showed differences between employees who left and those who stayed.
7. Outliers were detected using IQR and Z-score methods.
8. Visualizations were created to support the main findings.
""")

------------------------------------------------------------------------------------------------------------------
unsupervised
# ============================================================
# House Price Prediction
# Decision Tree, KNN, and Linear Regression
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PART 1 - DATA PREPROCESSING
# ============================================================

# Load dataset
df = pd.read_csv("house_price.csv")   # Change file name if needed

print("Dataset loaded successfully.")


# ------------------------------------------------------------
# 1. Dataset Overview
# ------------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nTotal Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst Five Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# 2. Missing Values
# ------------------------------------------------------------

print("\nMissing Values Before Handling:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Fill missing numerical values with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Handling:")
print(df.isnull().sum().sum())


# ------------------------------------------------------------
# 3. Drop HouseID if present
# ------------------------------------------------------------

if "HouseID" in df.columns:
    df = df.drop("HouseID", axis=1)
    print("\nHouseID column dropped.")


# ------------------------------------------------------------
# 4. Confirm Target Variable
# ------------------------------------------------------------

target = "SalePrice"

if target not in df.columns:
    raise ValueError("SalePrice column not found in dataset.")

print("\nTarget Variable:", target)


# ------------------------------------------------------------
# 5. Histograms for Numerical Features
# ------------------------------------------------------------

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numerical_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# ------------------------------------------------------------
# 6. Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(14, 10))
correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# ------------------------------------------------------------
# 7. Top 3 Features Correlated with SalePrice
# ------------------------------------------------------------

saleprice_corr = correlation_matrix[target].drop(target).abs().sort_values(ascending=False)

top_3_features = saleprice_corr.head(3).index.tolist()

print("\nTop 3 Features Most Correlated with SalePrice:")
print(top_3_features)

print("\nCorrelation Values:")
print(correlation_matrix[target][top_3_features])


# ------------------------------------------------------------
# 8. Scatter Plots for Top 3 Features
# ------------------------------------------------------------

for col in top_3_features:
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x=col, y=target)
    plt.title(f"{col} vs SalePrice")
    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.show()


# ------------------------------------------------------------
# 9. Boxplots for Outlier Detection
# ------------------------------------------------------------

for col in top_3_features + [target]:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, y=col)
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.show()


# ------------------------------------------------------------
# 10. Prepare Features and Target
# ------------------------------------------------------------

X = df.drop(target, axis=1)
y = df[target]

# Keep only numerical features
X = X.select_dtypes(include=["int64", "float64"])

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)


# ------------------------------------------------------------
# 11. Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nFeature scaling completed using StandardScaler.")


# ============================================================
# PART 2 - MODEL TRAINING
# ============================================================

# ------------------------------------------------------------
# 1. Train-Test Split: 75% Training, 25% Testing
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)


# ------------------------------------------------------------
# 2. Define Models
# ------------------------------------------------------------

knn_model = KNeighborsRegressor(
    n_neighbors=5,
    metric="euclidean"
)

decision_tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

linear_model = LinearRegression(
    fit_intercept=True
)


# ------------------------------------------------------------
# 3. Train Models
# ------------------------------------------------------------

knn_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
linear_model.fit(X_train, y_train)

print("\nAll models trained successfully.")


# ------------------------------------------------------------
# 4. Make Predictions
# ------------------------------------------------------------

knn_predictions = knn_model.predict(X_test)
tree_predictions = decision_tree_model.predict(X_test)
linear_predictions = linear_model.predict(X_test)


# ============================================================
# PART 3 - MODEL EVALUATION
# ============================================================

def evaluate_model(model_name, y_true, y_pred):
    """
    This function calculates regression evaluation metrics.
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }


# Evaluate all models
results = []

results.append(evaluate_model("KNN Regressor", y_test, knn_predictions))
results.append(evaluate_model("Decision Tree Regressor", y_test, tree_predictions))
results.append(evaluate_model("Linear Regression", y_test, linear_predictions))

results_df = pd.DataFrame(results)

print("\nModel Evaluation Results:")
print(results_df)


# ------------------------------------------------------------
# 1. Sort Models by R2 Score
# ------------------------------------------------------------

results_sorted = results_df.sort_values(by="R2 Score", ascending=False)

print("\nModels Sorted by R2 Score:")
print(results_sorted)


# ------------------------------------------------------------
# 2. Best Model
# ------------------------------------------------------------

best_model = results_sorted.iloc[0]

print("\nBest Performing Model:")
print(best_model)


# ------------------------------------------------------------
# 3. Bar Plot Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="R2 Score")
plt.title("Model Comparison Based on R2 Score")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.xticks(rotation=30)
plt.show()


plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="RMSE")
plt.title("Model Comparison Based on RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=30)
plt.show()


# ------------------------------------------------------------
# 4. Actual vs Predicted Plot for Each Model
# ------------------------------------------------------------

prediction_data = {
    "KNN Regressor": knn_predictions,
    "Decision Tree Regressor": tree_predictions,
    "Linear Regression": linear_predictions
}

for model_name, predictions in prediction_data.items():
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Actual vs Predicted SalePrice - {model_name}")
    
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    
    plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nFinal Summary:")
print("""
1. The dataset was loaded and basic information was displayed.
2. Missing numerical values were handled using median imputation.
3. HouseID was dropped because it is only an identifier.
4. Histograms, heatmap, scatter plots, and boxplots were created.
5. Top features correlated with SalePrice were identified.
6. Data was split into 75% training and 25% testing sets.
7. KNN, Decision Tree, and Linear Regression models were trained.
8. Models were evaluated using MAE, MSE, RMSE, and R2 Score.
9. The best model was selected based on the highest R2 Score and lowest error values.
""")

---------------------------------------------------------------------------------------------------------------------------------------------------
# Smart Traffic Incident Detection - Bayesian Network using pgmpy

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ==============================
# 1. Define Bayesian Network
# ==============================

model = DiscreteBayesianNetwork([
    ("CameraFault", "TrafficAlert"),
    ("RadarFault", "TrafficAlert"),
    ("LoopFault", "TrafficAlert"),
    ("TrafficAlert", "ControlRoomNotif"),
    ("TrafficAlert", "MobileAlert")
])


# ==============================
# 2. Define CPDs
# ==============================

# Prior probabilities
cpd_camera = TabularCPD(
    variable="CameraFault",
    variable_card=2,
    values=[
        [0.94],  # CameraFault = No
        [0.06]   # CameraFault = Yes
    ],
    state_names={"CameraFault": ["No", "Yes"]}
)

cpd_radar = TabularCPD(
    variable="RadarFault",
    variable_card=2,
    values=[
        [0.91],  # RadarFault = No
        [0.09]   # RadarFault = Yes
    ],
    state_names={"RadarFault": ["No", "Yes"]}
)

cpd_loop = TabularCPD(
    variable="LoopFault",
    variable_card=2,
    values=[
        [0.93],  # LoopFault = No
        [0.07]   # LoopFault = Yes
    ],
    state_names={"LoopFault": ["No", "Yes"]}
)


# CPT for TrafficAlert
# Evidence order: CameraFault, RadarFault, LoopFault
# Alert should be most likely when all three sensors fail.

cpd_alert = TabularCPD(
    variable="TrafficAlert",
    variable_card=2,
    values=[
        # TrafficAlert = No
        [0.99, 0.75, 0.70, 0.35, 0.65, 0.30, 0.25, 0.02],

        # TrafficAlert = Yes
        [0.01, 0.25, 0.30, 0.65, 0.35, 0.70, 0.75, 0.98]
    ],
    evidence=["CameraFault", "RadarFault", "LoopFault"],
    evidence_card=[2, 2, 2],
    state_names={
        "TrafficAlert": ["No", "Yes"],
        "CameraFault": ["No", "Yes"],
        "RadarFault": ["No", "Yes"],
        "LoopFault": ["No", "Yes"]
    }
)


# P(ControlRoomNotif | TrafficAlert)
cpd_control = TabularCPD(
    variable="ControlRoomNotif",
    variable_card=2,
    values=[
        [0.97, 0.03],  # ControlRoomNotif = No
        [0.03, 0.97]   # ControlRoomNotif = Yes
    ],
    evidence=["TrafficAlert"],
    evidence_card=[2],
    state_names={
        "ControlRoomNotif": ["No", "Yes"],
        "TrafficAlert": ["No", "Yes"]
    }
)


# P(MobileAlert | TrafficAlert)
cpd_mobile = TabularCPD(
    variable="MobileAlert",
    variable_card=2,
    values=[
        [0.96, 0.12],  # MobileAlert = No
        [0.04, 0.88]   # MobileAlert = Yes
    ],
    evidence=["TrafficAlert"],
    evidence_card=[2],
    state_names={
        "MobileAlert": ["No", "Yes"],
        "TrafficAlert": ["No", "Yes"]
    }
)


# ==============================
# 3. Add CPDs to Model
# ==============================

model.add_cpds(
    cpd_camera,
    cpd_radar,
    cpd_loop,
    cpd_alert,
    cpd_control,
    cpd_mobile
)


# ==============================
# 4. Validate Model
# ==============================

print("Model is valid:", model.check_model())


# ==============================
# 5. Perform Inference
# ==============================

inference = VariableElimination(model)


# Question 1:
# What is P(TrafficAlert = Yes | ControlRoomNotif = Yes, MobileAlert = Yes)?

query_1 = inference.query(
    variables=["TrafficAlert"],
    evidence={
        "ControlRoomNotif": "Yes",
        "MobileAlert": "Yes"
    }
)

print("\nQuestion 1:")
print("Probability of Traffic Alert given Control Room Notification and Mobile Alert:")
print(query_1)


# Extract exact value
prob_alert_given_notifications = query_1.values[1]
print("P(TrafficAlert = Yes | ControlRoomNotif = Yes, MobileAlert = Yes) =",
      round(prob_alert_given_notifications, 4))


# Question 2:
# What is P(RadarFault = Yes | TrafficAlert = Yes)?

query_2 = inference.query(
    variables=["RadarFault"],
    evidence={
        "TrafficAlert": "Yes"
    }
)

print("\nQuestion 2:")
print("Probability that Radar Sensor is Faulty given Traffic Alert:")
print(query_2)


# Extract exact value
prob_radar_fault_given_alert = query_2.values[1]
print("P(RadarFault = Yes | TrafficAlert = Yes) =",
      round(prob_radar_fault_given_alert, 4))

------------------------------------------------------------------------------------------------------------------
import math
AI = "X"
HUMAN = "O"
EMPTY = " "

def print_board(board):
    print()
    for i in range(3):
        print(" " + " | ".join(board[i]))
        if i < 2:
            print("---+---+---")
    print()

def check_winner(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != EMPTY:
            return row[0]

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != EMPTY:
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != EMPTY:
        return board[0][2]

    # Check draw
    for row in board:
        if EMPTY in row:
            return None

    return "Draw"

def evaluate(board):
    winner = check_winner(board)
    if winner == AI:
        return 1
    elif winner == HUMAN:
        return -1
    elif winner == "Draw":
        return 0
    return None

def alpha_beta_tictactoe(board, alpha, beta, maximizing_player):
    score = evaluate(board)
    if score is not None:
        return score
    if maximizing_player:
        best_score = -math.inf

        for row in range(3):
            for col in range(3):
                if board[row][col] == EMPTY:
                    board[row][col] = AI
                    score = alpha_beta_tictactoe(board, alpha, beta, False)
                    board[row][col] = EMPTY

                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)

                    if beta <= alpha:
                        return best_score
        return best_score
    else:
        best_score = math.inf
        for row in range(3):
            for col in range(3):
                if board[row][col] == EMPTY:
                    board[row][col] = HUMAN
                    score = alpha_beta_tictactoe(board, alpha, beta, True)
                    board[row][col] = EMPTY

                    best_score = min(best_score, score)
                    beta = min(beta, best_score)

                    if beta <= alpha:
                        return best_score
        return best_score

def best_ai_move(board):
    best_score = -math.inf
    best_move = None
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                board[row][col] = AI

                score = alpha_beta_tictactoe(
                    board,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=False
                )

                board[row][col] = EMPTY

                if score > best_score:
                    best_score = score
                    best_move = (row, col)
    return best_move

def human_move(board):
    while True:
        try:
            row, col = map(int, input("Enter row and column (0-2): ").split())

            if row < 0 or row > 2 or col < 0 or col > 2:
                print("Invalid range. Use values from 0 to 2.")
                continue

            if board[row][col] != EMPTY:
                print("Cell already occupied. Try again.")
                continue

            board[row][col] = HUMAN
            break

        except ValueError:
            print("Invalid input. Example: 0 2")

def play_game():
    board = [
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]
    print("Tic-Tac-Toe with Alpha-Beta Pruning")
    print("Human = O")
    print("AI = X")
    print_board(board)

    while True:
        human_move(board)
        print_board(board)
        result = check_winner(board)
        if result is not None:
            break
        print("AI is thinking...")
        move = best_ai_move(board)
        if move is not None:
            row, col = move
            board[row][col] = AI
        print_board(board)
        result = check_winner(board)
        if result is not None:
            break

    if result == AI:
        print("AI wins!")
    elif result == HUMAN:
        print("Human wins!")
    else:
        print("Draw!")
play_game()
-------------------------------------------------------------------------------
"""
1. Word Game using Alpha-Beta Pruning
Logic
Max tries to form the best valid word.
Min tries to reduce Max's final score.
Alpha-Beta Pruning skips useless branches.

Scoring:

Valid word score   = length of word
Invalid word score = -5
"""

import math

VALID_WORDS = {
    "cat", "bat", "bad", "dog", "good", "hat",
    "rat", "car", "card", "cart", "go", "to",
    "art", "tag", "god", "bag", "bar"
}

MAX_PLAYER = "Max"
MIN_PLAYER = "Min"
INVALID_WORD_PENALTY = -5


def is_valid_word(word):
    return word.lower() in VALID_WORDS


def has_possible_valid_word_prefix(prefix):
    prefix = prefix.lower()
    for word in VALID_WORDS:
        if word.startswith(prefix):
            return True
    return False


def evaluate_word(word):
    word = word.lower()
    if is_valid_word(word):
        return len(word)
    return INVALID_WORD_PENALTY


def is_terminal(letters, current_word, max_length):
    if len(letters) == 0:
        return True
    if len(current_word) >= max_length:
        return True
    if is_valid_word(current_word):
        return True
    if current_word != "" and not has_possible_valid_word_prefix(current_word):
        return True
    return False


def alpha_beta_word_game(letters, current_word, depth, alpha, beta, maximizing_player, max_length):
    if depth == 0 or is_terminal(letters, current_word, max_length):
        return evaluate_word(current_word)

    if maximizing_player:
        best_score = -math.inf

        for i in range(len(letters)):
            chosen_letter = letters[i]
            new_letters = letters[:i] + letters[i + 1:]
            new_word = current_word + chosen_letter

            score = alpha_beta_word_game(
                new_letters,
                new_word,
                depth - 1,
                alpha,
                beta,
                False,
                max_length
            )

            best_score = max(best_score, score)
            alpha = max(alpha, best_score)

            if beta <= alpha:
                break

        return best_score

    best_score = math.inf

    for i in range(len(letters)):
        chosen_letter = letters[i]
        new_letters = letters[:i] + letters[i + 1:]
        new_word = current_word + chosen_letter

        score = alpha_beta_word_game(
            new_letters,
            new_word,
            depth - 1,
            alpha,
            beta,
            True,
            max_length
        )

        best_score = min(best_score, score)
        beta = min(beta, best_score)

        if beta <= alpha:
            break

    return best_score


def best_move_for_max(letters, current_word, depth, max_length):
    best_score = -math.inf
    best_letter = None
    best_index = None

    for i in range(len(letters)):
        chosen_letter = letters[i]
        new_letters = letters[:i] + letters[i + 1:]
        new_word = current_word + chosen_letter

        score = alpha_beta_word_game(
            new_letters,
            new_word,
            depth - 1,
            -math.inf,
            math.inf,
            False,
            max_length
        )

        if score > best_score:
            best_score = score
            best_letter = chosen_letter
            best_index = i

    return best_letter, best_index, best_score


def best_move_for_min(letters, current_word, depth, max_length):
    best_score = math.inf
    best_letter = None
    best_index = None

    for i in range(len(letters)):
        chosen_letter = letters[i]
        new_letters = letters[:i] + letters[i + 1:]
        new_word = current_word + chosen_letter

        score = alpha_beta_word_game(
            new_letters,
            new_word,
            depth - 1,
            -math.inf,
            math.inf,
            True,
            max_length
        )

        if score < best_score:
            best_score = score
            best_letter = chosen_letter
            best_index = i

    return best_letter, best_index, best_score


def play_word_game():
    letters = ["c", "a", "t", "b", "d", "o", "g"]
    current_word = ""
    max_length = 4
    depth = 4
    turn = MAX_PLAYER

    print("Word Game using Alpha-Beta Pruning")
    print("Valid Words:", sorted(VALID_WORDS))
    print("Available Letters:", letters)
    print()

    while letters and not is_terminal(letters, current_word, max_length):
        print("Current Word:", current_word if current_word else "_")
        print("Remaining Letters:", letters)

        if turn == MAX_PLAYER:
            letter, index, score = best_move_for_max(letters, current_word, depth, max_length)
            print(f"Max chooses: {letter}")
            current_word += letter
            letters.pop(index)
            turn = MIN_PLAYER
        else:
            letter, index, score = best_move_for_min(letters, current_word, depth, max_length)
            print(f"Min chooses: {letter}")
            current_word += letter
            letters.pop(index)
            turn = MAX_PLAYER

        print()

    final_score = evaluate_word(current_word)

    print("Final Word:", current_word)

    if is_valid_word(current_word):
        print("Result: Valid word")
        print("Score:", final_score)
    else:
        print("Result: Invalid word")
        print("Penalty:", final_score)


play_word_game()

---------------------------------------------------------------------------------------
import math


def alpha_beta_coins(coins, left, right, max_score, min_score, alpha, beta, maximizing_player):
    if left > right:
        return max_score - min_score

    if maximizing_player:
        best_value = -math.inf

        # Pick left coin
        value_left = alpha_beta_coins(
            coins,
            left + 1,
            right,
            max_score + coins[left],
            min_score,
            alpha,
            beta,
            False
        )

        best_value = max(best_value, value_left)
        alpha = max(alpha, best_value)

        if beta <= alpha:
            return best_value

        # Pick right coin
        value_right = alpha_beta_coins(
            coins,
            left,
            right - 1,
            max_score + coins[right],
            min_score,
            alpha,
            beta,
            False
        )

        best_value = max(best_value, value_right)
        alpha = max(alpha, best_value)

        return best_value

    else:
        best_value = math.inf

        # Pick left coin
        value_left = alpha_beta_coins(
            coins,
            left + 1,
            right,
            max_score,
            min_score + coins[left],
            alpha,
            beta,
            True
        )

        best_value = min(best_value, value_left)
        beta = min(beta, best_value)

        if beta <= alpha:
            return best_value

        # Pick right coin
        value_right = alpha_beta_coins(
            coins,
            left,
            right - 1,
            max_score,
            min_score + coins[right],
            alpha,
            beta,
            True
        )

        best_value = min(best_value, value_right)
        beta = min(beta, best_value)

        return best_value


def best_move_for_max(coins, left, right, max_score, min_score):
    value_if_left = alpha_beta_coins(
        coins,
        left + 1,
        right,
        max_score + coins[left],
        min_score,
        -math.inf,
        math.inf,
        False
    )

    value_if_right = alpha_beta_coins(
        coins,
        left,
        right - 1,
        max_score + coins[right],
        min_score,
        -math.inf,
        math.inf,
        False
    )

    if value_if_left >= value_if_right:
        return "left"
    else:
        return "right"


def play_coin_game(coins):
    left = 0
    right = len(coins) - 1

    max_score = 0
    min_score = 0

    print("Initial Coins:", coins)

    while left <= right:
        # Max turn
        move = best_move_for_max(coins, left, right, max_score, min_score)

        if move == "left":
            picked = coins[left]
            left += 1
        else:
            picked = coins[right]
            right -= 1

        max_score += picked
        print(f"Max picks {picked}, Remaining Coins: {coins[left:right + 1]}")

        if left > right:
            break

        # Min turn
        # This version lets Min greedily choose the smaller edge coin
        if coins[left] <= coins[right]:
            picked = coins[left]
            left += 1
        else:
            picked = coins[right]
            right -= 1

        min_score += picked
        print(f"Min picks {picked}, Remaining Coins: {coins[left:right + 1]}")

    print("\nFinal Scores")
    print("Max:", max_score)
    print("Min:", min_score)

    if max_score > min_score:
        print("Winner: Max")
    elif min_score > max_score:
        print("Winner: Min")
    else:
        print("Draw")


coins = [3, 9, 1, 2, 7, 5]
play_coin_game(coins)

-----------------------------------------------------------------------------------------------
