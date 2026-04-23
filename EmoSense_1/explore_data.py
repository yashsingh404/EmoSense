"""
FER2013 Dataset Explorer
=========================
Visualize sample images from each emotion class and
check class distribution before training.

Run: python explore_data.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48
DATA_PATH = "data/fer2013.csv"
PLOTS_DIR = "model/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    images, labels, usages = [], [], []
    for _, row in df.iterrows():
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        images.append(pixels)
        labels.append(int(row['emotion']))
        usages.append(row.get('Usage', 'Training'))
    return np.array(images), np.array(labels), np.array(usages)


def plot_samples(images, labels):
    """Show one sample per emotion class."""
    fig, axes = plt.subplots(1, 7, figsize=(18, 3))
    fig.patch.set_facecolor('#0d0d1a')
    fig.suptitle('FER2013 — Sample Images per Class', color='white', fontsize=14, y=1.02)

    for i, emotion in enumerate(EMOTIONS):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            axes[i].set_visible(False); continue
        axes[i].imshow(images[idx[0]], cmap='gray')
        axes[i].set_title(emotion, color='white', fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    out = f"{PLOTS_DIR}/samples.png"
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved samples → {out}")
    plt.close()


def plot_distribution(labels, usages):
    """Bar chart of class distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#0d0d1a')
    splits = [('Training', 'Training'), ('PublicTest', 'Validation'), ('PrivateTest', 'Test')]
    colors = ['#7c6bff', '#00e5ff', '#ffd166', '#ff4d4d', '#4a9eff', '#ff9f1c', '#94a3b8']

    for ax, (usage_key, title) in zip(axes, splits):
        mask = usages == usage_key
        counts = [np.sum(labels[mask] == i) for i in range(len(EMOTIONS))]
        bars = ax.bar(EMOTIONS, counts, color=colors)
        ax.set_facecolor('#0d0d1a')
        ax.set_title(title, color='white', fontsize=12)
        ax.tick_params(colors='#aaa'); ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
        for spine in ax.spines.values(): spine.set_color('#333')
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(count), ha='center', fontsize=8, color='#ccc')

    plt.suptitle('Class Distribution Across Splits', color='white', fontsize=14)
    plt.tight_layout()
    out = f"{PLOTS_DIR}/class_distribution.png"
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved distribution → {out}")
    plt.close()


def print_stats(labels, usages):
    print("\n=== Dataset Statistics ===")
    for usage_key, title in [('Training','Train'), ('PublicTest','Val'), ('PrivateTest','Test')]:
        mask = usages == usage_key
        total = mask.sum()
        print(f"\n{title} ({total} samples):")
        for i, emo in enumerate(EMOTIONS):
            n = np.sum(labels[mask] == i)
            pct = n/total*100 if total else 0
            bar = '█' * int(pct/2)
            print(f"  {emo:10s}: {n:5d}  ({pct:5.1f}%)  {bar}")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Download FER2013 from https://www.kaggle.com/datasets/msambare/fer2013")
        return

    print("Loading dataset...")
    images, labels, usages = load_data(DATA_PATH)
    print(f"Total samples: {len(images)}")

    print_stats(labels, usages)
    plot_samples(images, labels)
    plot_distribution(labels, usages)
    print("\nDone! Check model/plots/ for visuals.")


if __name__ == "__main__":
    main()
