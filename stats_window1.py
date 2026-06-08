import re

def analyze_log(logfile, start_day=101, end_day=120):
    try:
        with open(logfile, 'r', encoding='utf-16') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

    current_trades = []
    current_pnl = 0.0
    current_equity = 0.0
    
    trades_in_window = []
    daily_pnls = []
    
    # Regexes
    trade_pattern = re.compile(r'([A-Z0-9.]+)\s*\|\s*AI=\d+\.\d+\s*\|\s*W=\d+\.\d+%\s*\|\s*(.*?)\s*\|\s*[+]?([-\d.]+)')
    pnl_pattern = re.compile(r'PnL:\s*.*?\s*([-\d.]+)\s*\(.*?\) \| Capital: ([\d.]+)')
    day_pattern = re.compile(r'\[WF Day (\d+)\]\s*(\d+) \|\s*Equity:\s*([\d.]+)')

    for line in lines:
        # Strip ansi color codes
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
        
        if '| AI=' in clean_line:
            parts = clean_line.split('|')
            if len(parts) >= 5:
                # 603933.SH | AI=0.101 | W=33.9% | T+1 5% stop-loss | -1858.11
                code = parts[0].strip()
                ai_score = parts[1].strip()
                weight = parts[2].strip()
                exit_type = parts[3].strip()
                pnl_val = float(parts[4].strip())
                current_trades.append({
                    'code': code, 'exit': exit_type, 'pnl': pnl_val
                })
        elif clean_line.startswith('PnL:'):
            match = pnl_pattern.search(clean_line)
            if match:
                current_pnl = float(match.group(1))
        elif '[WF Day ' in clean_line:
            match = day_pattern.search(clean_line)
            if match:
                day_num = int(match.group(1))
                date_str = match.group(2)
                equity = float(match.group(3))
                
                if start_day <= day_num <= end_day:
                    trades_in_window.extend(current_trades)
                    daily_pnls.append({
                        'day': day_num, 'date': date_str, 'pnl': current_pnl, 'equity': equity
                    })
                
                # Reset for next day
                current_trades = []
                current_pnl = 0.0

    # Calculate statistics
    if not daily_pnls:
        print(f"No data found for Window {start_day} - {end_day}")
        return

    # Trade stats
    total_trades = len(trades_in_window)
    winning_trades = [t for t in trades_in_window if t['pnl'] > 0]
    losing_trades = [t for t in trades_in_window if t['pnl'] <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    exit_counts = {}
    for t in trades_in_window:
        exit_type = t['exit']
        exit_counts[exit_type] = exit_counts.get(exit_type, 0) + 1

    start_equity = 1000000.0  # approximate base
    end_equity = daily_pnls[-1]['equity']
    window_return = (end_equity / start_equity) - 1.0

    print("="*50)
    print(f"Window 1 Statistics (Day {start_day} to {end_day})")
    print(f"Period: {daily_pnls[0]['date']} to {daily_pnls[-1]['date']}")
    print("="*50)
    print(f"Starting Equity : {start_equity:.2f}")
    print(f"Ending Equity   : {end_equity:.2f}")
    print(f"Window Return   : {window_return*100:.2f}%")
    print("-" * 50)
    print(f"Total Trades    : {total_trades}")
    print(f"Win Rate        : {win_rate*100:.2f}%")
    print(f"Winning Trades  : {len(winning_trades)}")
    print(f"Losing Trades   : {len(losing_trades)}")
    print("-" * 50)
    print("Exit Reasons Distribution:")
    for ext, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"  {ext}: {count}")
    print("="*50)

if __name__ == '__main__':
    analyze_log('wf_backtest_final.log')
