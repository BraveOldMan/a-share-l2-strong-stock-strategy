import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

csv_path_top1 = r'models/wf_equity_curve.csv'
csv_path_top3 = r'models/wf_equity_curve_top3.csv'
csv_path_top5 = r'v19_archive/wf_equity_curve.csv'
out_path = r'C:/Users/MrLee/.gemini/antigravity/brain/21186a49-70e7-4f7d-ac0e-29c8d654e8b9/v19_triad_comparison.png'

try:
    df1 = pd.read_csv(csv_path_top1)
    df1['date'] = pd.to_datetime(df1['date'].astype(str), format='%Y%m%d')
    
    df3 = pd.read_csv(csv_path_top3)
    df3['date'] = pd.to_datetime(df3['date'].astype(str), format='%Y%m%d')
    
    df5 = pd.read_csv(csv_path_top5)
    df5['date'] = pd.to_datetime(df5['date'].astype(str), format='%Y%m%d')

    plt.figure(figsize=(13, 7))
    plt.plot(df1['date'], df1['capital'], color='#2ca02c', linewidth=2.0, alpha=0.9, label='ALL-IN Top-1 (max_positions=1)')
    plt.plot(df3['date'], df3['capital'], color='#d62728', linewidth=2.5, alpha=1.0, label='Concentrated Top-3 (max_positions=3)')
    plt.plot(df5['date'], df5['capital'], color='#1f77b4', linewidth=1.8, alpha=0.9, label='Classic Top-5 (max_positions=5)')

    plt.title('V19 Concentration Triad: All-in vs Concentrated vs Classic', fontsize=15, pad=15)
    plt.xlabel('Date (OOS)', fontsize=12)
    plt.ylabel('Capital', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    plt.axhline(y=1000000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    plt.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print('Triad plot saved successfully.')
except Exception as e:
    print(f'Failed to plot: {e}')
