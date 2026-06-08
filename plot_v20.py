import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

csv_path_v20 = r'models/wf_equity_curve.csv'
csv_path_v19_top3 = r'models/wf_equity_curve_top3.csv'
out_path = r'C:/Users/MrLee/.gemini/antigravity/brain/21186a49-70e7-4f7d-ac0e-29c8d654e8b9/v20_vs_v19_top3.png'

try:
    df_v20 = pd.read_csv(csv_path_v20)
    df_v20['date'] = pd.to_datetime(df_v20['date'].astype(str), format='%Y%m%d')
    
    df_v19 = pd.read_csv(csv_path_v19_top3)
    df_v19['date'] = pd.to_datetime(df_v19['date'].astype(str), format='%Y%m%d')

    plt.figure(figsize=(12, 6))
    plt.plot(df_v20['date'], df_v20['capital'], color='#ff7f0e', linewidth=2.0, alpha=0.9, label='V20 Upgraded Top-3 (ATR/Sector/Fire-Alarm)')
    plt.plot(df_v19['date'], df_v19['capital'], color='#d62728', linewidth=2.5, alpha=1.0, label='V19 Pristine Top-3 (Raw ML Signal)')

    plt.title('The Price of Over-Engineering: V20 vs V19 pristine Top-3', fontsize=14, pad=15)
    plt.xlabel('Date (OOS)', fontsize=12)
    plt.ylabel('Capital', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    plt.axhline(y=1000000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print('Plot saved successfully.')
except Exception as e:
    print(f'Failed to plot: {e}')
