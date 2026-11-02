
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Compute absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'creditcard.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'notebooks', 'plots')

# Add src to path for potential custom imports
sys.path.append(PROJECT_ROOT)

def perform_eda():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Please run data_ingestion.py first.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating EDA plots...")

    try:
        # 1. Class Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Class', data=df)
        plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
        plot_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {plot_path}")

        # 2. Time vs Amount (Log Scale)
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
        bins = 50
        ax1.hist(df.Amount[df.Class == 1], bins=bins, color='red', alpha=0.7)
        ax1.set_title('Fraud')
        ax1.set_yscale('log') # Log scale for better visibility

        ax2.hist(df.Amount[df.Class == 0], bins=bins, color='blue', alpha=0.5)
        ax2.set_title('Normal')
        ax2.set_yscale('log')

        plt.xlabel('Amount ($)')
        plt.ylabel('Number of Transactions (Log Scale)')
        plot_path = os.path.join(OUTPUT_DIR, "amount_distribution_by_class.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {plot_path}")

        # 3. Correlation Matrix (Subsample)
        plt.figure(figsize=(12, 10))
        # Use subsample for speed and clarity if large
        if len(df) > 10000:
            sub_sample_corr = df.sample(10000, random_state=42).corr()
        else:
            sub_sample_corr = df.corr()
            
        sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size': 20})
        plt.title('Correlation Matrix (Subsampled)', fontsize=14)
        plot_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {plot_path}")

    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    perform_eda()
