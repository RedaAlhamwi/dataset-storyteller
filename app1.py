from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# ✅ Initialize OpenAI client
client = OpenAI(
    api_key="sk-...."
)
# ✅ Function: Basic EDA summary with numeric & categorical features
def basic_eda_summary(df):
    summary = {}

    summary["Shape"] = df.shape
    summary["Columns"] = list(df.columns)
    summary["Data Types"] = df.dtypes.astype(str).to_dict()

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()
    summary["Missing Values"] = missing

    # Numeric stats
    numeric_df = df.select_dtypes(include="number")
    stats = numeric_df.describe().T[["mean", "std", "min", "max"]].round(2).to_dict()
    summary["Key Statistics"] = stats

    # Top correlations
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr().abs()
        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
            .head(3)
            .to_dict()
        )
        summary["Top Correlations"] = {f"{k[0]} & {k[1]}": v for k, v in corr_pairs.items()}

    # Categorical analysis
    categorical_df = df.select_dtypes(include=["object", "category"])
    cat_summary = {}
    for col in categorical_df.columns:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            most_common = mode[0]
            freq = df[col].value_counts(normalize=True)[most_common]
            cat_summary[col] = {
                "Most Frequent": most_common,
                "Frequency (%)": round(freq * 100, 2)
            }
    summary["Categorical Summary"] = cat_summary

    return summary

# ✅ Create prompt for LLM
def create_prompt(summary):
    return f"""
You are a data storytelling assistant. Convert the following EDA summary into a beginner-friendly explanation. Highlight:
- Dataset structure
- Noteworthy features
- Missing values
- Any patterns, trends, or correlations

Use simple language and write in 2–3 short paragraphs.

EDA Summary:
{summary}
"""

# ✅ Plot 1: Missing Values
def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None
    plt.figure(figsize=(8, 4))
    sns.barplot(x=missing.values, y=missing.index, palette="viridis")
    plt.title("Missing Values per Column")
    plt.xlabel("Count")
    plt.ylabel("Column")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)

# ✅ Plot 2: Correlation Heatmap
def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)

# ✅ Main handler function
def analyze_csv(file):
    try:
        df = pd.read_csv(file.name)
        summary = basic_eda_summary(df)
        prompt = create_prompt(summary)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        story = response.choices[0].message.content

        # Generate plots
        plot1 = plot_missing_values(df)
        plot2 = plot_correlation_heatmap(df)

        return story, plot1, plot2

    except Exception as e:
        return f"❌ Error: {e}", None, None

# ✅ Gradio app
interface = gr.Interface(
    fn=analyze_csv,
    inputs=gr.File(label="📂 Upload your CSV file"),
    outputs=[
        gr.Textbox(label="📖 AI-generated EDA Summary"),
        gr.Image(label="📉 Missing Values Plot"),
        gr.Image(label="📊 Correlation Heatmap")
    ],
    title="📊 Dataset Storyteller: EDA Narrator",
    description="Upload a dataset and get a friendly AI-written summary and visualizations of its structure and trends."
)

interface.launch()

