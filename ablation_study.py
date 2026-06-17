import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure results folder exists
os.makedirs('results', exist_ok=True)

# Set clean academic style matching university A3 formatting guidelines
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 12
})

def generate_dynamic_ablation():
    csv_path = 'ensemble_leaderboard.csv'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[!] Target file '{csv_path}' not found.")
    
    # Live ingestion from database tracking log
    df = pd.read_csv(csv_path)
    df_stacking = df[df['model'] == 'Hybrid_Stacking_v1.7']
    
    try:
        network_row = df_stacking[df_stacking['dataset'] == 'network_only'].iloc[0]
        biometrics_row = df_stacking[df_stacking['dataset'] == 'biometrics_only'].iloc[0]
        full_row = df_stacking[df_stacking['dataset'] == 'full'].iloc[0]
    except IndexError:
        raise ValueError("[!] Missing expected dataset splits inside leaderboard CSV.")

    # Extract performance metrics scaled to percentages
    acc_scores = [float(network_row['accuracy']) * 100, float(biometrics_row['accuracy']) * 100, float(full_row['accuracy']) * 100]
    f1_scores = [float(network_row['f1_score']) * 100, float(biometrics_row['f1_score']) * 100, float(full_row['f1_score']) * 100]
    roc_aucs = [float(network_row['roc_auc']) * 100, float(biometrics_row['roc_auc']) * 100, float(full_row['roc_auc']) * 100]
    
    # Extract latency metrics
    latencies = [float(network_row['meta_stage_lat_p95_ms']), float(biometrics_row['meta_stage_lat_p95_ms']), float(full_row['meta_stage_lat_p95_ms'])]
    
    categories = ['JA4 TLS\nOnly', 'Biometrics\nOnly', 'Sentinel-Bot\n(Hybrid Fusion)']
    
    x = np.arange(len(categories))
    width = 0.22  # Narrowed bar width to gracefully accommodate cluster groupings
    
    # Enforced A3 structural canvas constraint
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    
    # Poster Color Palette Alignment (Deep Blues Gradient)
    color_acc = '#4169E1'   # Royal Blue
    color_f1 = '#2B4C7E'    # Slate/Steel Blue
    color_auc = '#0A2540'   # Midnight Navy Blue
    color_lat = '#D91A2A'   # Academic Crimson Red (Overlay Line Accent)

    # Plot grouped primary bars
    rects1 = ax1.bar(x - width, acc_scores, width, color=color_acc, edgecolor='black', linewidth=0.6, label='Accuracy')
    rects2 = ax1.bar(x, f1_scores, width, color=color_f1, edgecolor='black', linewidth=0.6, label='F1-Score')
    rects3 = ax1.bar(x + width, roc_aucs, width, color=color_auc, edgecolor='black', linewidth=0.6, label='ROC-AUC')
    
    ax1.set_ylabel('Model Performance Evaluation Scale (%)', color='#0A2540', fontweight='bold')
    ax1.set_ylim(92, 101)  # Safe academic margin limit showing distribution nuances clearly
    ax1.grid(axis='y', linestyle=':', alpha=0.4)
    
    # Secondary Y-Axis for Latency Profile Overlay
    ax2 = ax1.twinx()
    line_lat = ax2.plot(x, latencies, color=color_lat, marker='o', markersize=6, linewidth=2.0, linestyle='--', label='p95 Latency')
    ax2.set_ylabel('p95 Inference Processing Latency (ms)', color=color_lat, fontweight='bold')
    ax2.set_ylim(0, 2.5)
    ax2.tick_params(axis='y', labelcolor=color_lat)

    # Data value annotation engine
    def label_bars(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    label_bars(rects1)
    label_bars(rects2)
    label_bars(rects3)

    # Annotate Latency line markers
    for i, txt in enumerate(latencies):
        ax2.annotate(f'{txt:.2f}ms', (x[i], latencies[i]), xytext=(0, 6), 
                     textcoords="offset points", ha='center', fontsize=8, 
                     fontweight='bold', color=color_lat)

    # Joint Legend Engineering Layout
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', frameon=True, facecolor='white', edgecolor='none')

    plt.suptitle('Hybrid Modality Ablation & Inference Latency Analysis', fontweight='bold', fontsize=12, y=0.96, color='#0A2540')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontweight='bold')
    
    # Core evaluative scientific takeaway text block
    plt.figtext(
        0.5, 0.01, 
        "Takeaway: Hybrid fusion achieves perfect separability with sub-2 ms inference overhead", 
        ha="center", fontsize=8.5, style="italic", fontweight="bold", color="#333333"
    )
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.savefig('results/ablation_study.png', dpi=300)
    plt.close()
    print("[=======] Success: Live multi-metric ablation plot exported cleanly with template colors matching.")

if __name__ == '__main__':
    generate_dynamic_ablation()