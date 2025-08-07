# 📊 Dataset Storyteller: EDA Narrator

An AI-powered application that transforms your raw CSV data into beginner-friendly insights. Built as part of the **Building AI Applications Challenge** by [Decoding Data Science](https://www.linkedin.com/groups/9371112/), this project integrates Python, OpenAI's GPT API, and Gradio for a seamless and interactive user experience.

---

## 🚀 Project Overview

**Dataset Storyteller: EDA Narrator** allows users to upload a CSV file and instantly receive:
- Descriptive statistics (shape, columns, data types)
- Missing values summary
- Correlation matrix for numerical features
- Frequency analysis for categorical columns
- Human-readable summary generated via OpenAI's GPT

---

## 🧠 Key Features

- 🔎 **Exploratory Data Analysis (EDA)**: Automatic detection of numeric and categorical features.
- 🧠 **LLM Integration**: Sends a compact EDA summary to OpenAI API and receives beginner-friendly explanations.
- 📈 **Visuals**: Plots top categorical feature frequencies.
- 🌐 **Web App**: Built using Gradio for a fast and interactive browser-based interface.

---

## 🧰 Tech Stack

- **Python**
- **Gradio** (UI)
- **Pandas & Matplotlib** (EDA & plotting)
- **OpenAI GPT API** (text generation)
- **dotenv** (API key management)

---

## 🧪 How to Run the App

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/eda-narrator.git
   cd eda-narrator
