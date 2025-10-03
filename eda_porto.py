import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")
    return df


def save_text(content: str, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def basic_overview(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=== Shape ===")
    lines.append(str(df.shape))
    lines.append("")

    lines.append("=== Dtypes ===")
    lines.append(str(df.dtypes))
    lines.append("")

    lines.append("=== Head (first 5 rows) ===")
    lines.append(str(df.head()))
    lines.append("")

    lines.append("=== Missing values per column ===")
    missing = df.isna().sum().sort_values(ascending=False)
    lines.append(str(missing))
    lines.append("")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    lines.append("=== Numeric columns ===")
    lines.append(", ".join(num_cols) if num_cols else "<none>")
    lines.append("")

    if num_cols:
        lines.append("=== Describe (numeric) ===")
        lines.append(str(df[num_cols].describe(include="all")))

    return "\n".join(lines)


def correlation_heatmap(df: pd.DataFrame, out_file: str, max_cols: int = 40) -> Optional[str]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        return "No numeric columns for correlation heatmap."
    if num_df.shape[1] > max_cols:
        num_df = num_df.iloc[:, :max_cols]

    plt.figure(figsize=(min(24, 1 + num_df.shape[1] * 0.6), min(18, 1 + num_df.shape[1] * 0.6)))
    corr = num_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    return None


def distributions(df: pd.DataFrame, out_dir: str, max_cols: int = 20) -> str:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return "No numeric columns for distributions."

    cols_to_plot = num_cols[:max_cols]
    for col in cols_to_plot:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dist_{col}.png"), dpi=200)
        plt.close()
    if len(num_cols) > max_cols:
        return f"Plotted first {max_cols} of {len(num_cols)} numeric columns."
    return ""


def pairplot_sample(df: pd.DataFrame, out_file: str, max_cols: int = 6, sample_size: int = 1000) -> str:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if len(num_cols) < 2:
        return "Not enough numeric columns for pairplot."

    sample_df = df[num_cols].dropna()
    if sample_df.shape[0] > sample_size:
        sample_df = sample_df.sample(sample_size, random_state=42)

    g = sns.pairplot(sample_df, corner=True, diag_kind="hist")
    g.fig.suptitle("Pairplot (sample)", y=1.02)
    plt.tight_layout()
    g.savefig(out_file, dpi=200)
    plt.close()
    return ""


def run_eda(csv_path: str, out_dir: str = "eda_reports") -> None:
    ensure_output_dir(out_dir)
    ensure_output_dir(os.path.join(out_dir, "figures"))

    df = load_csv(csv_path)

    # Overview text
    overview = basic_overview(df)
    save_text(overview, os.path.join(out_dir, "overview.txt"))

    # Correlation heatmap
    corr_msg = correlation_heatmap(df, os.path.join(out_dir, "figures", "correlation_heatmap.png"))

    # Distributions
    dist_msg = distributions(df, os.path.join(out_dir, "figures"))

    # Pairplot
    pair_msg = pairplot_sample(df, os.path.join(out_dir, "figures", "pairplot.png"))

    # Notes file for any messages
    notes = []
    for msg in [corr_msg, dist_msg, pair_msg]:
        if msg:
            notes.append(msg)
    if notes:
        save_text("\n".join(notes), os.path.join(out_dir, "notes.txt"))

    print(f"EDA complete. Reports saved to: {out_dir}")


if __name__ == "__main__":
    # Default to local porto.csv in project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(project_root, "porto.csv")
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else default_csv
    run_eda(csv_arg)


