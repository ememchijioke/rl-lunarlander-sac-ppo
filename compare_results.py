# compare_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt

from config import PLOTS_DIR

RESULTS = [
    {
        "Algorithm": "PPO",
        "Training Steps": "1,000,000",
        "Mean Reward": 195.38,
        "Best Reward": 266.63,
        "Worst Reward": 25.41,
        "Notes": "Good baseline; inconsistent landing.",
    },
    {
        "Algorithm": "Custom SAC",
        "Training Steps": "1,000,000",
        "Mean Reward": 233.57,
        "Best Reward": 282.58,
        "Worst Reward": 16.45,
        "Notes": "9/10 strong episodes.",
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

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    image_path = os.path.join(PLOTS_DIR, "evaluation_summary_table.png")
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)

    print(f"Saved table image: {image_path}")


if __name__ == "__main__":
    main()
