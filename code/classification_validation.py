from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# %% Load training data
data = pd.read_csv(r"C:\Users\isabe\Downloads\BRCA_merged.csv", index_col=0)

stage_map = {
    "T1": 0, "T1a": 0, "T1b": 0, "T1c": 0,
    "T2": 0, "T2a": 0, "T2b": 0, "T2c": 0,
    "T3": 1, "T3a": 1, "T3b": 1, "T3c": 1,
    "T4": 1, "T4a": 1, "T4b": 1, "T4c": 1
}
data["stage_binary"] = data["ajcc_tumor_pathologic_pt"].map(stage_map)
data = data.dropna(subset=["stage_binary"])

meta_cols = ["ajcc_tumor_pathologic_pt", "cancer_type", "stage_binary"]
gene_cols = [col for col in data.columns if col not in meta_cols]
y = data["stage_binary"].values

# %% Set best features
feature_1 = "FZD3"
feature_2 = "CACNA1D"
X_tr = data[[feature_1, feature_2]].values
y_tr = data["stage_binary"].values

y_label = [{0: "Early Stage (I/II)", 1: "Late Stage (III/IV)"}[i] for i in y_tr]

# %% Scatter plot
sns.scatterplot(x=X_tr[:, 0], y=X_tr[:, 1], hue=y_label, palette="Set1")
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title("Training Data")
plt.show()

# %% DECISION TREE
dt_model = DecisionTreeClassifier(max_depth=2).fit(X_tr, y_tr)
print("Decision Tree Training accuracy:", dt_model.score(X_tr, y_tr))

plt.figure()
plot_tree(dt_model, feature_names=[feature_1, feature_2],
          class_names=["Early Stage", "Late Stage"], filled=True)
plt.show()

# %% LOGISTIC REGRESSION
model = LogisticRegression(penalty=None, max_iter=1000).fit(X_tr, y_tr)
print("Logistic Regression Training accuracy:", model.score(X_tr, y_tr))

# Decision boundary plot
x_min, x_max = X_tr[:, 0].min(), X_tr[:, 0].max()
y_min, y_max = X_tr[:, 1].min(), X_tr[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
sns.scatterplot(x=X_tr[:, 0], y=X_tr[:, 1], hue=y_label,
                edgecolors='k', palette="Set1", alpha=0.8)
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title("Logistic Regression Decision Boundary")
plt.show()

# %% Load validation dataset
BRCA_gene_data = pd.read_csv(
    r"C:\Users\isabe\Downloads\VALIDATION_SET_GSE62944_subsample_log2TPM.csv", index_col=0, header=0)
metadata_df = pd.read_csv(
    r'C:\Users\isabe\OneDrive\Documents\BME2315\Module-4-Cancer-Grosso-Orlando\Grosso-Orlando-Module-4-Cancer\data\VALIDATION_SET_GSE62944_metadata.csv', index_col=0, header=0)

cancer_samples = metadata_df[metadata_df['cancer_type'] == 'BRCA'].index
BRCA_data = BRCA_gene_data[cancer_samples]

with open(r"C:\Users\isabe\OneDrive\Documents\BME2315\Module-4-Cancer-Grosso-Orlando\Grosso-Orlando-Module-4-Cancer\code\Menyhart_JPA_CancerHallmarks_core.txt", "r") as f:
    lines = f.readlines()

sp_line = lines[8].split("\t")
desired_gene_list = [gene.strip() for gene in sp_line if gene.strip() not in ["SUSTAINING PROLIFERATIVE SIGNALING", ""]]
desired_gene_list = list(set(desired_gene_list))
gene_list = [gene for gene in desired_gene_list if gene in BRCA_data.index]
BRCA_gene_data_filtered = BRCA_data.loc[gene_list]

BRCA_metadata = metadata_df.loc[cancer_samples]
BRCA_merged_val = BRCA_gene_data_filtered.T.merge(BRCA_metadata, left_index=True, right_index=True)

BRCA_merged_val["stage_binary"] = BRCA_merged_val["ajcc_tumor_pathologic_pt"].map(stage_map)
BRCA_merged_val = BRCA_merged_val.dropna(subset=["stage_binary"])

X_val = BRCA_merged_val[[feature_1, feature_2]].values
y_val = BRCA_merged_val["stage_binary"].values

# %% Compare both models on validation set
print("\n--- RESULTS ---")
print(f"Decision Tree       — Train: {dt_model.score(X_tr, y_tr):.3f}  Val: {dt_model.score(X_val, y_val):.3f}")
print(f"Logistic Regression — Train: {model.score(X_tr, y_tr):.3f}  Val: {model.score(X_val, y_val):.3f}")

# %% Confusion matrices for both
for name, m in [("Decision Tree", dt_model), ("Logistic Regression", model)]:
    y_pred = m.predict(X_val)
    print(f"\n{name}:")
    print(classification_report(y_val, y_pred, target_names=["Early Stage", "Late Stage"]))
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Early Stage", "Late Stage"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name} (Validation Set)")
    plt.show()
