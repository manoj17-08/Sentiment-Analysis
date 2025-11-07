#  sentiment-analysis-amazon-reviews 🛍️

## Project Overview
This project delivers an **end-to-end Natural Language Processing (NLP) pipeline** for performing sentiment analysis on a dataset of Amazon Fine Food Reviews. The primary goal is to compare the performance and efficacy of two distinct sentiment analysis methodologies: a traditional lexicon-based model and a state-of-the-art transformer-based deep learning model.

This project demonstrates strong proficiency in **Exploratory Data Analysis (EDA), machine learning model implementation, and performance evaluation**—key skills for Data Science and Machine Learning roles.

---

## ✨ Features and Key Techniques

This repository implements the full lifecycle of a data science project, focusing on the following techniques:

| Category | Technique Implemented | Description |
| :--- | :--- | :--- |
| **Traditional NLP** | **VADER (Valence Aware Dictionary and sEntiment Reasoner)** | Utilized a lexicon-based model for rapid sentiment scoring, demonstrating a baseline understanding of ad-hoc text analysis. |
| **Modern NLP** | **RoBERTa (Robustly Optimized BERT Pre-training Approach)** | Implemented a **pre-trained Hugging Face Transformer model** for sequence classification, showcasing mastery of deep learning and context-aware NLP. |
| **Deep Learning** | **Transfer Learning** | Applied the fine-tuned RoBERTa model to the Amazon review dataset, significantly improving accuracy over the baseline without extensive training time. |
| **Analysis & Validation** | **Model Comparison & Edge Cases** | Conducted visual and statistical comparison (using Compound scores) between VADER and RoBERTa, highlighting performance differences, particularly in handling nuanced or sarcastic text. |
| **Workflow** | **Jupyter Notebook Pipeline** | Code is structured within a reproducible Kaggle/Jupyter Notebook environment for clear, cell-by-cell execution and documentation. |

---

## 🛠️ Technology Stack & Prerequisites

The project is built entirely in Python and relies on the following major libraries:

* **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn` (for EDA and visualization).
* **Traditional NLP:** `nltk` (specifically the VADER lexicon).
* **Deep Learning NLP:** `transformers` (Hugging Face library for RoBERTa and Pipeline usage).
* **Tools:** Jupyter Notebook or Kaggle Notebook environment (recommended for GPU access).

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/pik1989/Sentiment-Analysis.git](https://github.com/pik1989/Sentiment-Analysis.git)
    cd Sentiment-Analysis
    ```
2.  **Create Environment (Recommended):**
    ```bash
    conda create -n sentiment_env python=3.9
    conda activate sentiment_env
    ```
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn nltk transformers torch
    ```

---

## 📊 Sample Performance Insights

The primary value of the project is the comparison of how the models score the sentiment (Positive, Neutral, Negative) against the user's actual 1-5 star review score:

| Metric | VADER (Lexicon-Based) | RoBERTa (Transformer) | Conclusion |
| :--- | :--- | :--- | :--- |
| **Confidence in Prediction** | Lower (Scores closer to 0.0, higher Neutral) | **Higher** (Scores often near 0.9 for Positive/Negative) | RoBERTa is **more decisive** due to its contextual understanding. |
| **Handling Sarcasm** | **Poor.** Often misclassifies 1-star reviews if they contain positive words (e.g., "The package arrived quickly, but the product was garbage"). | **Good.** Able to capture the overall negative context despite conflicting keywords. |
| **Correlation to 5-Star Reviews** | Moderate positive trend. | **Strong positive trend.** | RoBERTa shows a clearer correlation between the review text and the explicit star rating. |

---

## 🚀 Usage

The full project workflow is contained within the main Jupyter Notebook.

1.  **Open the Notebook:** Launch your Jupyter environment (`jupyter notebook`) and open `Sentiment_Analysis_Project.ipynb`.
2.  **Data Loading:** Ensure the necessary dataset (`reviews.csv`) is accessible in the same directory or properly mounted if using a Kaggle/Colab environment.
3.  **Execution:** Run the cells sequentially to perform the following steps:
    * Load and clean the raw data.
    * Perform **VADER** scoring and initial EDA.
    * Load the Hugging Face **RoBERTa** model and tokenizer.
    * Run the deep learning model across the entire dataset (Note: **GPU runtime is highly recommended** for fast RoBERTa execution).
    * Generate the final comparative visualizations and review text analysis.

---

## 📧 Contact & License

* **Author/Maintainer:** [Your Name]
* **Contact:** [Your Email Address]
* **License:** This project is licensed under the **MIT License**.
