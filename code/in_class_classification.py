#Claude AI was used to assist with applying the code to our data, as well as cleaning the dataset and debugging the code.
# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# %%
# Load the merged BRCA dataset
data = pd.read_csv(r"C:\Users\isabe\Downloads\BRCA_merged.csv", index_col=0)

#Map tumor stages to early (I-II) and late (III-IV)
stage_map = {
    "T1":    0, "T1a":   0, "T1b":   0, "T1c":  0,
    "T2":   0, "T2a":  0, "T2b":  0, "T2c": 0,
    "T3":  1, "T3a": 1, "T3b": 1, "T3c": 1,
    "T4":   1, "T4a":   1, "T4b":    1, "T4c": 1
}
data["stage_binary"] = data["ajcc_tumor_pathologic_pt"].map(stage_map)
data = data.dropna(subset=["stage_binary"]) #remove samples with missing tumor stage

#Separate features (gene expression) from labels (tumor stage)
meta_cols = ["ajcc_tumor_pathologic_pt", "cancer_type", "stage_binary"] 
gene_cols = [col for col in data.columns if col not in meta_cols]

X = data[gene_cols].values #feature matrix
y = data["stage_binary"].values #target labels (0 = early, 1=late)

# %%
#Convert binary labels to stage names
y_label = [{0: "Early Stage (I/II)", 1: "Late Stage (III/IV)"}[i] for i in y]
feature_1 = "CALML5"
feature_2 = "ESR1"
X = data[[feature_1, feature_2]].values
sns.scatterplot(x=X[:, 0],
                y=X[:, 1],
                hue=y_label,
                palette="Set1")


# %%
# Logistic regression

# BUILD A MODEL: 
model = LogisticRegression(penalty=None).fit(X, y)

# PREDICT AND EVALUATE: 
model.predict_proba(X)
print(model.score(X, y))

# %% Plotting decision boundary

# Create meshgrid
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Compute decision function over the grid
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)  # background
plt.contour(xx, yy, Z, levels=[0], colors='black',
            linewidths=2)  # decision boundary
sns.scatterplot(x=X[:, 0],
                y=X[:, 1],
                hue=y_label,
                edgecolors='k',
                palette="Set1",
                alpha=0.8)
plt.legend()
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title("Logistic Regression Decision Boundary")
plt.show()

# %% DECISION TREE CLASSIFIER
# BUILD A MODEL: 
dt_model = DecisionTreeClassifier(max_depth=3).fit(X, y)
# PREDICT AND EVALUATE: 
print(dt_model.score(X, y))
# %% PLOT DECISION TREE
plot_tree(dt_model, feature_names=[
          feature_1, feature_2], class_names=["Early Stage", "Late Stage"], filled=True)
# %% TRY TO BUILD A BETTER CLASSIFIER BY PICKING BETTER FEATURES!
# to do this, you can loop over all pairs of features in the dataset
