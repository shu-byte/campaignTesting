# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap

st.set_page_config(layout="wide")
st.title("🎯 Credit Card Campaign AI & Clustering App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("CC GENERAL.csv")
    df = df.dropna()
    return df

df = load_data()

# Simulate target
df['accepted_offer'] = (
    (df['BALANCE'] > 1500) &
    (df['PURCHASES_FREQUENCY'] > 0.5) &
    (df['PRC_FULL_PAYMENT'] > 0.5)
).astype(int)

# Split X and y
X = df.drop(columns=["CUST_ID", "accepted_offer"])
y = df['accepted_offer']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# SHAP explainability
st.subheader("🔍 Feature Importance (SHAP)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For binary classification or regression models
fig_shap = shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight')

# Clustering section
st.subheader("🧩 Customer Clustering (PCA + KMeans)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Plot clusters
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["cluster"] = clusters

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="Set2", ax=ax)
st.pyplot(fig)

# Cluster summary
df["cluster"] = clusters
st.subheader("📊 Cluster Profiles")
st.dataframe(df.groupby("cluster").mean(numeric_only=True).round(2))
