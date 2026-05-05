from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

meta_cols = ["ajcc_tumor_pathologic_pt", "cancer_type", "stage_binary",
             "DSS.time", "OS.time", "PFI.time", "DFI.time", "DFI", "PFI", "OS", "DSS",
             "age_at_diagnosis"]
gene_cols = [col for col in data.columns
             if col not in meta_cols and pd.api.types.is_numeric_dtype(data[col])]

y = data["stage_binary"].values

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

# %% FIX 1: Find features shared between BOTH datasets before selecting top genes
# This is why you only had 3 genes before — now you'll have many more
shared_genes = [g for g in gene_cols if g in BRCA_merged_val.columns]
print(f"Shared genes available: {len(shared_genes)}")

early = data[data["stage_binary"] == 0][shared_genes]
late  = data[data["stage_binary"] == 1][shared_genes]
diff  = (late.mean() - early.mean()).abs().sort_values(ascending=False)

# Use top 20 shared genes instead of top 2
top_features = diff.index[:20].tolist()

print(f"Top features selected: {top_features}")

# %% Prepare train/val arrays
X_tr = data[top_features].fillna(0).values
y_tr = data["stage_binary"].values

X_val = BRCA_merged_val[top_features].fillna(0).values
y_val = BRCA_merged_val["stage_binary"].values



# %% FIX 2: Scale features — critical for Logistic Regression
scaler = StandardScaler()
X_tr_scaled  = scaler.fit_transform(X_tr)   # fit on train only
X_val_scaled = scaler.transform(X_val)       # apply same scale to val

# %% DECISION TREE
# FIX 3: Lower max_depth to reduce overfitting (was 5, now 3)
dt_model = DecisionTreeClassifier(max_depth=3, class_weight='balanced',
                                   random_state=0).fit(X_tr, y_tr)

plt.figure(figsize=(20, 8))
plot_tree(dt_model, feature_names=top_features,
          class_names=["Early Stage", "Late Stage"], filled=True, fontsize=8)
plt.title("Decision Tree (max_depth=3)")
plt.tight_layout()
plt.show()

# %% LOGISTIC REGRESSION
# FIX 4: Add L2 regularization (C=0.1) + use scaled data
# %% LOGISTIC REGRESSION
# Use only 2 features for decision boundary plot
plot_features = top_features[:2]
X_tr_2d = data[plot_features].fillna(0).values
X_val_2d = BRCA_merged_val[plot_features].fillna(0).values  # add this line

model = LogisticRegression(penalty=None, max_iter=1000, class_weight='balanced').fit(X_tr_2d, y_tr)
print(f"Logistic Regression Training accuracy: {model.score(X_tr_2d, y_tr):.3f}")
print(f"Logistic Regression Validation accuracy: {model.score(X_val_2d, y_val):.3f}")

# Decision boundary plot
x_min, x_max = X_tr_2d[:, 0].min(), X_tr_2d[:, 0].max()
y_min, y_max = X_tr_2d[:, 1].min(), X_tr_2d[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

y_label = ["Late Stage" if val == 1 else "Early Stage" for val in y_tr]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
sns.scatterplot(x=X_tr_2d[:, 0], y=X_tr_2d[:, 1], hue=y_label,
                edgecolors='k', palette="Set1", alpha=0.8)
plt.xlabel(plot_features[0])
plt.ylabel(plot_features[1])
plt.title("Logistic Regression Decision Boundary")
plt.tight_layout()
plt.show()
print(f"Decision Tree Training accuracy: {dt_model.score(X_tr, y_tr):.3f}")
print(f"Decision Tree Validation accuracy: {dt_model.score(X_val, y_val):.3f}")
# %% Results

# Logistic Regression results
y_pred_lr = model.predict(X_val_2d)
print("Logistic Regression:")
print(classification_report(y_val, y_pred_lr, target_names=["Early Stage", "Late Stage"]))
cm = confusion_matrix(y_val, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Early Stage", "Late Stage"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Logistic Regression (Validation Set)")
plt.tight_layout()
plt.show()

# Decision Tree results
y_pred_dt = dt_model.predict(X_val)
print("Decision Tree:")
print(classification_report(y_val, y_pred_dt, target_names=["Early Stage", "Late Stage"]))
cm = confusion_matrix(y_val, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Early Stage", "Late Stage"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Decision Tree (Validation Set)")
plt.tight_layout()
plt.show()

print(f"Logistic Regression Training accuracy: {model.score(X_tr_2d, y_tr):.3f}")
print(f"Logistic Regression Validation accuracy: {model.score(X_val_2d, y_val):.3f}")


print(f"Decision Tree Training accuracy: {dt_model.score(X_tr, y_tr):.3f}")
print(f"Decision Tree Validation accuracy: {dt_model.score(X_val, y_val):.3f}")
