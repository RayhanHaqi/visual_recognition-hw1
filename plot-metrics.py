import pandas as pd
import matplotlib.pyplot as plt

def plot_training_metrics(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File {csv_file} not found!")
        return

    if 'Epoch' not in df.columns:
        print("Error: Column 'Epoch' not found in data.")
        return

    epochs = df['Epoch']

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Training & Validation Metrics', fontsize=16, fontweight='bold')

    ax1 = axes[0]
    if 'T_Loss' in df.columns and 'V_Loss' in df.columns:
        ax1.plot(epochs, df['T_Loss'], label='Training Loss', color='tab:red', marker='o')
        ax1.plot(epochs, df['V_Loss'], label='Validation Loss', color='tab:orange', marker='s')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = axes[1]
    if 'Top1_Acc' in df.columns and 'Top5_Acc' in df.columns:
        ax2.plot(epochs, df['Top1_Acc'], label='Top-1 Accuracy (%)', color='tab:blue', marker='o')
        ax2.plot(epochs, df['Top5_Acc'], label='Top-5 Accuracy (%)', color='tab:cyan', marker='s')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Top-1 vs Top-5 Accuracy')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

    ax3 = axes[2]
    metrics_cols = ['F1_Score', 'Precision', 'Recall']
    colors = ['tab:green', 'tab:purple', 'tab:brown']
    markers = ['o', 's', '^']
    
    for col, color, marker in zip(metrics_cols, colors, markers):
        if col in df.columns:
            ax3.plot(epochs, df[col], label=f'{col} (%)', color=color, marker=marker)
            
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Score (%)')
    ax3.set_title('F1-Score, Precision, and Recall')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'training_curves.png'")
    
    plt.show()

if __name__ == "__main__":
    plot_training_metrics('/home/tilakoid/selectedtopics/cv_hw1_data/logs/distill_resnetrs200.tf_in1k-convnext_xxlarge.clip_laion2b_soup_ft_in1k.csv')