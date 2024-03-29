# Import libraries
import pandas as pd
import numpy as np
import warnings
import shap
# Data Visualization
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

# Configure libraries
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 10)
plt.style.use("seaborn")

# Load dataset
df_bank = pd.read_csv("bank.csv")

# Drop 'duration' column
df_bank = df_bank.drop("duration", axis=1)

# print(df_bank.info())
print("Shape of dataframe:", df_bank.shape)
df_bank.head()

# Copying original dataframe
df_bank_ready = df_bank.copy()

conditions = [
    (df_bank['day'] <= 10),
    (df_bank['day'] > 10) & (df_bank['day'] < 20),
    (df_bank['day'] >= 20)]
choices = ['beginning of month', 'middle of month', 'end of month']
df_bank['month_period'] = np.select(conditions, choices, default='unknown')
df_bank.head()

conditions = [
    (df_bank['month'] == 'dec') | (df_bank['month'] == 'jan') | (df_bank['month'] == 'feb'),
    (df_bank['month'] == 'mar') | (df_bank['month'] == 'apr') | (df_bank['month'] == 'may'),
    (df_bank['month'] == 'jun') | (df_bank['month'] == 'jul') | (df_bank['month'] == 'aug'),
    (df_bank['month'] == 'sep') | (df_bank['month'] == 'oct') | (df_bank['month'] == 'nov')
]
choices = ['winter', 'summer', 'autumn','spring']
df_bank['season'] = np.select(conditions, choices, default='unknown')
df_bank.head()

scaler = StandardScaler()
num_cols = ["age", "balance", "day", "campaign", "pdays", "previous"]
df_bank_ready[num_cols] = scaler.fit_transform(df_bank[num_cols])

encoder = OneHotEncoder(sparse=False)
cat_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Encode Categorical Data
df_encoded = pd.DataFrame(encoder.fit_transform(df_bank_ready[cat_cols]))
df_encoded.columns = encoder.get_feature_names_out(cat_cols)

# Replace Categotical Data with Encoded Data
df_bank_ready = df_bank_ready.drop(cat_cols, axis=1)
df_bank_ready = pd.concat([df_encoded, df_bank_ready], axis=1)

# Encode target value
df_bank_ready["deposit"] = df_bank_ready["deposit"].apply(
    lambda x: 1 if x == "yes" else 0
)

print("Shape of dataframe:", df_bank_ready.shape)

# Select Features
feature = df_bank_ready.drop("deposit", axis=1)

# Select Target
target = df_bank_ready["deposit"]

# Set Training and Testing Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    feature, target, shuffle=True, test_size=0.2, random_state=1
)

# Show the Training and Testing Data
print("Shape of training feature:", X_train.shape)
print("Shape of testing feature:", X_test.shape)
print("Shape of training label:", y_train.shape)
print("Shape of training label:", y_test.shape)


def evaluate_model(model, x_test, y_test):
    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "kappa": kappa,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "cm": cm,
    }


# Building Decision Tree model
dtc = tree.DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)

# Evaluate Model
dtc_eval = evaluate_model(dtc, X_test, y_test)

# Print result
print("Accuracy:", dtc_eval["acc"])
print("Precision:", dtc_eval["prec"])
print("Recall:", dtc_eval["rec"])
print("F1 Score:", dtc_eval["f1"])
print("Cohens Kappa Score:", dtc_eval["kappa"])
print("Area Under Curve:", dtc_eval["auc"])
print("Confusion Matrix:\n", dtc_eval["cm"])

# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor("white")

# First plot
# set bar size
barWidth = 0.2
dtc_score = [
    dtc_eval["acc"],
    dtc_eval["prec"],
    dtc_eval["rec"],
    dtc_eval["f1"],
    dtc_eval["kappa"],
]

# Set position of bar on X axis
r1 = np.arange(len(dtc_score))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plot
ax1.bar(r1, dtc_score, width=barWidth, edgecolor="white", label="Decision Tree")

# Configure x and y axis
ax1.set_xlabel("Metrics", fontweight="bold")
labels = ["Accuracy", "Precision", "Recall", "F1", "Kappa"]
ax1.set_xticks(
    [r + (barWidth * 1.5) for r in range(len(dtc_score))],
)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Score", fontweight="bold")
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title("Evaluation Metrics", fontsize=14, fontweight="bold")
ax1.legend()

# Second plot
## Plotting the ROC curve
#ax2.plot(
#    dtc_eval["fpr"],
#    dtc_eval["tpr"],
#    label="Decision Tree, auc = {:0.5f}".format(dtc_eval["auc"]),
#)

## Configure x and y axis
#ax2.set_xlabel("False Positive Rate", fontweight="bold")
#ax2.set_ylabel("True Positive Rate", fontweight="bold")

## Create legend & title
#ax2.set_title("ROC Curve", fontsize=14, fontweight="bold")
#ax2.legend(loc=4)

#plt.show()

#making a PR curve
model_dt = DecisionTreeClassifier().fit(X_train,y_train)
probs_dt = model_dt.predict_proba(X_test)[:,1]
precision_dt, recall_dt,thresholds = precision_recall_curve(y_test,probs_dt)
auc_dt = auc(recall_dt,precision_dt)
ax2.plot(recall_dt,precision_dt,label=f"AUC (Decision Tree) ={auc_dt:.2f}")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("PR Curve")
plt.show()

# Prepares a default instance of the random forest regressor
model = RandomForestRegressor()
# Fits the model on the data
model.fit(X_train, y_train)
# Fits the explainer
explainer = shap.Explainer(model.predict, X_test)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
