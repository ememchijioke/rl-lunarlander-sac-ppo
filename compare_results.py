# compare_results.py

import os

import matplotlib.pyplot as plt
import pandas as pd

from config import PLOTS_DIR


RESULTS = [
    {
        "Algorithm": "Custom Deep PPO",
        "Role": "Main model",
        "Steps": "1,000,000",
        "Mean": 220.07,
        "Best": 284.26,
        "Worst": 36.26,
    },
    {
        "Algorithm": "Custom SAC",
        "Role": "Comparison",
        "Steps": "1,000,000",
        "Mean": 233.57,
        "Best": 282.58,
        "Worst": 16.45,
    },
]


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.DataFrame(RESULTS)

    csv_path = os.path.join(PLOTS_DIR, "evaluation_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\nEvaluation Summary")
    print("------------------")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)

    image_path = os.path.join(PLOTS_DIR, "evaluation_summary_table.png")
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)

    print(f"Saved table image: {image_path}")


if __name__ == "__main__":
    main()