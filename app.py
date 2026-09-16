import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer t-SNE Dashboard", layout="wide")

st.title("Customer Segmentation & t-SNE Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

st.subheader("Raw Data")
st.write(df.head())

# Preprocessing
le_gender = LabelEncoder()
df["GenderEnc"] = le_gender.fit_transform(df["Gender"])

feature_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "GenderEnc"]
X = df[feature_cols].copy()
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE
st.sidebar.header("t-SNE Settings")
perplexity = st.sidebar.slider("Perplexity", 5, 50, 20, 5)
n_iter = st.sidebar.slider("t-SNE iterations", 250, 1000, 400, 50)
random_state = st.sidebar.number_input("Random state", value=42)

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    n_iter=n_iter,
    random_state=random_state,
    learning_rate="auto",
    init="pca",
    method="barnes_hut"
)
X_tsne = tsne.fit_transform(X_scaled)
df["tSNE_1"] = X_tsne[:, 0]
df["tSNE_2"] = X_tsne[:, 1]

# Clustering
st.sidebar.header("Clustering Settings")
n_clusters = st.sidebar.slider("Number of clusters (K)", 2, 10, 5, 1)

kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
df["Segment"] = kmeans.fit_predict(X_scaled)

# Filters
st.sidebar.header("Filters")

seg_options = sorted(df["Segment"].unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Select segments",
    options=seg_options,
    default=seg_options
)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
income_min, income_max = int(df["Annual Income (k$)").min(), int(df["Annual Income (k$)").max())

age_range = st.sidebar.slider(
    "Age range",
    int(age_min),
    int(age_max),
    (int(age_min), int(age_max))
)

income_range = st.sidebar.slider(
    "Annual Income range (k$)",
    int(income_min),
    int(income_max),
    (int(income_min), int(income_max))
)

mask = (
    df["Segment"].isin(selected_segments) &
    df["Age"].between(age_range[0], age_range[1]) &
    df["Annual Income (k$)").between(income_range[0], income_range[1])
)
df_filt = df[mask].copy()

# t-SNE plot
st.subheader("2D t-SNE Visualization (Colored by Segment)")

fig_tsne = px.scatter(
    df_filt,
    x="tSNE_1",
    y="tSNE_2",
    color="Segment",
    hover_data=["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"],
    title="Customers in t-SNE Space",
    color_continuous_scale="Viridis",
)
fig_tsne.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
st.plotly_chart(fig_tsne, use_container_width=True)

# Summary stats
st.subheader("Summary Statistics per Segment")

summary = (
    df_filt.groupby("Segment")
    .agg(
        Count=("CustomerID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Annual Income (k$)", "mean"),
        Avg_Spending_Score=("Spending Score (1-100)", "mean"),
    )
    .reset_index()
)
st.dataframe(summary.style.format({
    "Avg_Age": "{:.1f}",
    "Avg_Income": "{:.1f}",
    "Avg_Spending_Score": "{:.1f}",
}))

# Additional charts
st.subheader("Feature Distributions by Segment")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.box(
        df_filt,
        x="Segment",
        y="Age",
        color="Segment",
        title="Age Distribution by Segment",
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_income = px.box(
        df_filt,
        x="Segment",
        y="Annual Income (k$)",
        color="Segment",
        title="Income Distribution by Segment",
    )
    st.plotly_chart(fig_income, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig_spend_hist = px.histogram(
        df_filt,
        x="Spending Score (1-100)",
        color="Segment",
        title="Spending Score Distribution",
        barmode="overlay",
        opacity=0.7,
    )
    st.plotly_chart(fig_spend_hist, use_container_width=True)

with col4:
    fig_income_hist = px.histogram(
        df_filt,
        x="Annual Income (k$)",
        color="Segment",
        title="Income Distribution",
        barmode="overlay",
        opacity=0.7,
    )
    st.plotly_chart(fig_income_hist, use_container_width=True)

st.subheader("Filtered Customer Data")
st.dataframe(df_filt.drop(columns=["GenderEnc", "tSNE_1", "tSNE_2"]))