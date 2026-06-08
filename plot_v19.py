import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

csv_path = r'v19_archive/wf_equity_curve.csv'
out_path = r'C:/Users/MrLee/.gemini/antigravity/brain/21186a49-70e7-4f7d-ac0e-29c8d654e8b9/v19_equity_curve.png'

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['capital'], color='#1f77b4', linewidth=2)
plt.title('V19 Alpha Engine - Walk-Forward Equity Curve (Out of Sample)', fontsize=14, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Capital', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# format x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

# plot benchmark/start line
plt.axhline(y=1000000, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
plt.legend()

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Successfully saved curve to {out_path}')
