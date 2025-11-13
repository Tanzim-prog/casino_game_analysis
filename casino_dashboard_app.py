import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# ==========================
# 1. File Configuration
# ==========================

input_path = r"D:\Project\Casino Game Project\github_files\Casino_Gaming_Data new.csv"
output_path = os.path.join(os.path.dirname(input_path), "casino_game_dataset.csv")

# ==========================
# 2. Load and Clean Dataset
# ==========================

@st.cache_data
def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    # Convert data types
    if "month_ending" in df.columns:
        df["month_ending"] = pd.to_datetime(df["month_ending"], errors="coerce")
    if "fiscal_year" in df.columns:
        df["fiscal_year"] = df["fiscal_year"].astype(str)

    for col in df.columns:
        if col not in ["fiscal_year", "month_ending", "licensee"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)

    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].fillna("Unknown")

    df = df.dropna(subset=["month_ending"])
    df.to_csv(output_path, index=False)
    return df

df = load_and_clean_data(input_path)

# ==========================
# 3. Streamlit Dashboard UI
# ==========================

st.set_page_config(page_title="Casino Gaming Data Dashboard", layout="wide")
st.title("🎰 Casino Gaming Data Dashboard")
st.markdown("This dashboard shows cleaned data insights, trends, and correlations from the casino gaming dataset.")

# ==========================
# 4. Overview Section
# ==========================

st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Unique Licensees", df['licensee'].nunique())

st.dataframe(df.head(10))

with st.expander("Show Summary Statistics"):
    st.write(df.describe())

# ==========================
# 5. Visual EDA
# ==========================

st.header("📈 Visual Exploratory Data Analysis")

# Sidebar Filters
st.sidebar.header("🔍 Filters")
licensee_filter = st.sidebar.multiselect(
    "Select Licensee(s)", options=df["licensee"].unique(), default=df["licensee"].unique()
)
filtered_df = df[df["licensee"].isin(licensee_filter)]

# 5.1 Total Gross Gaming Revenue Distribution
st.subheader("Distribution of Total Gross Gaming Revenue")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(filtered_df["total_gross_gaming_revenue"], kde=True, ax=ax1)
ax1.set_xlabel("Revenue")
st.pyplot(fig1)

# 5.2 Licensee vs Total Gross Gaming Revenue
st.subheader("Total Gross Gaming Revenue by Licensee")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(data=filtered_df, x="licensee", y="total_gross_gaming_revenue", estimator=sum, ci=None, ax=ax2)
plt.xticks(rotation=45, ha="right")
st.pyplot(fig2)

# 5.3 Correlation Heatmap
st.subheader("Correlation Heatmap (Numeric Columns)")
fig3, ax3 = plt.subplots(figsize=(10, 6))
corr = filtered_df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# 5.4 Monthly Trend by Licensee
st.subheader("Monthly Trend of Total Gross Gaming Revenue by Licensee")
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=filtered_df.sort_values("month_ending"), x="month_ending", y="total_gross_gaming_revenue", hue="licensee", ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)

# 5.5 Outlier Detection (Boxplot)
st.subheader("Outlier Detection - Total Gross Gaming Revenue")
fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.boxplot(data=filtered_df, x="total_gross_gaming_revenue", ax=ax5)
st.pyplot(fig5)

# ==========================
# 6. Insights Section
# ==========================

st.header("🧠 Key Insights")
total_revenue = filtered_df["total_gross_gaming_revenue"].sum()
highest_earner = (
    filtered_df.groupby("licensee")["total_gross_gaming_revenue"].sum().idxmax()
)
most_recent_month = filtered_df["month_ending"].max().strftime("%B %Y")

st.markdown(f"""
- **Total Gross Gaming Revenue:** ${total_revenue:,.0f}  
- **Top Performing Licensee:** {highest_earner}  
- **Most Recent Month in Data:** {most_recent_month}  
""")

# ==========================
# 7. Export Option
# ==========================

st.download_button(
    label="📥 Download Cleaned Dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="casino_game_dataset.csv",
    mime="text/csv",
)
