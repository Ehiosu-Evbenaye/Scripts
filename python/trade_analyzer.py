# trade_analyzer_for_MultiPair_3d.py
# FULLY UPDATED & ROBUST VERSION FOR MultiPair_3d EA
# - Robust CSV loading with fallbacks (comma → tab → auto-sniff)
# - Parses EA_TradeLog.csv correctly
# - Handles confluence extraction robustly
# - Matches on deal_ticket (or position_id if you add it to EA log)

import pandas as pd
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import csv

# ==================== CONFIGURATION ====================
# UPDATE THESE PATHS TO MATCH YOUR SYSTEM
trades_csv = r'C:\ALL PYTHON PROJECTS\Trading Performance Analyzer Tools\Backtest Data and Logs\Backtest_Deals_Export.csv'
log_csv = r'C:\Users\YourUsername\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EA_TradeLog.csv'  # <-- CHANGE TO YOUR ACTUAL PATH

# ==================== ROBUST CSV LOADER WITH FALLBACKS ====================
def load_deals_csv(path):
    print(f"Loading deals from: {path}")
    
    # Try 1: Default pandas behaviour (comma for .csv)
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:  # Reasonable number of columns
            print(f"Loaded successfully with comma separator ({df.shape[1]} columns detected).")
            return df
    except Exception as e:
        print(f"Comma attempt failed: {e}")
    
    # Try 2: Explicit tab separator
    try:
        df = pd.read_csv(path, sep='\t')
        if df.shape[1] > 1:
            print(f"Loaded successfully with tab separator ({df.shape[1]} columns detected).")
            return df
    except Exception as e:
        print(f"Tab attempt failed: {e}")
    
    # Try 3: Use csv.Sniffer to auto-detect delimiter
    try:
        with open(path, 'r', encoding='utf-8') as f:
            sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', '\t', ';'])
            f.seek(0)
            df = pd.read_csv(f, delimiter=dialect.delimiter)
            print(f"Loaded successfully with sniffed delimiter '{dialect.delimiter}' ({df.shape[1]} columns detected).")
            return df
    except Exception as e:
        print(f"Sniffer attempt failed: {e}")
    
    raise ValueError("Could not load the CSV file with any supported separator. Check the file format.")

# ==================== PARSER FOR EA_TradeLog.csv ====================
def parse_log_file(log_csv_path):
    decisions = {}
    try:
        df_log = pd.read_csv(log_csv_path, header=None, names=['full_line'])
        print(f"Loaded {len(df_log)} lines from EA_TradeLog.csv")
        
        for _, row in df_log.iterrows():
            line = row['full_line']
            if 'TRADE_ENTRY' not in line:
                continue
                
            fields = [f.strip() for f in line.split(',')]
            parts = {}
            conf_str = None
            
            for field in fields:
                if '=' in field:
                    k, v = field.split('=', 1)
                    parts[k] = v
                else:
                    if field.startswith('conf1='):
                        conf_str = field
            
            if conf_str:
                clean = conf_str.replace(' ', '')
                conf_matches = re.findall(r'conf(\d+)=(\d)', clean) # Already catches 1-11 automatically
                for num, val in conf_matches:
                    parts[f'conf{num}'] = int(val)
            
            if 'atr' in parts:
                parts['atr'] = float(parts['atr'])
            if 'daily_range_pips' in parts:
                parts['vol_elevation'] = float(parts['daily_range_pips'])
            
            # === NEW SMART MATCHING LOGIC (REPLACE HERE) ===
            ticket_key = None
            # Prefer position_id if present (this is the reliable match to export "Ticket")
            if 'position_id' in parts:
                ticket_key = parts['position_id']
            elif 'deal_ticket' in parts:
                # Fallback for old logs that only have deal_ticket
                ticket_key = parts['deal_ticket']
                print(f"Warning: Using fallback deal_ticket for matching (symbol={parts.get('symbol','?')}), consider adding position_id to EA logs for accuracy")
            
            if ticket_key:
                decisions[ticket_key] = parts
                
    except Exception as e:
        print(f"Error reading/parsing EA_TradeLog.csv: {e}")
    
    print(f"Parsed {len(decisions)} TRADE_ENTRY records from log.")
    return decisions

# ==================== ENRICHMENT ====================
def load_data(df, decisions):
    print("\nStarting enrichment...")
    print(f"Retrieved {len(decisions)} decision records from log.")
    
    pattern_columns = [
        'conf1', 'conf2', 'conf3', 'conf4', 'conf5', 'conf6', 'conf7', 'conf8', 'conf9', 'conf10',
        'atr', 'vol_elevation'
    ]
    
    for col in pattern_columns:
        df[col] = np.nan
    
    enriched_count = 0
    for idx, row in df.iterrows():
        ticket = str(row['Ticket'])
        if ticket in decisions:
            dec_data = decisions[ticket]
            for col in pattern_columns:
                if col in dec_data:
                    df.at[idx, col] = dec_data[col]
            enriched_count += 1
    
    print(f"Enriched {enriched_count}/{len(df)} trades with confluence/ATR/vol data.")
    if enriched_count == 0 and len(decisions) > 0:
        print("WARNING: No matches! Likely deal_ticket ≠ position ID.")
        print("   → Add position_id to TRADE_ENTRY log in EA for perfect matching.")
    
    # Time features
    df['Time'] = pd.to_datetime(df['Time'])
    df['Close Time'] = pd.to_datetime(df['Close Time'])
    df['Duration'] = (df['Close Time'] - df['Time']).dt.total_seconds() / 3600
    df['DayOfWeek'] = df['Time'].dt.day_name()
    df['Month'] = df['Time'].dt.month_name()
    df['Hour'] = df['Time'].dt.hour
    

    def get_session(h):
        if 0 <= h < 8:   return 'Asian'
        elif 8 <= h < 13: return 'London'
        elif 13 <= h < 17: return 'London_NY_Overlap'
        elif 17 <= h < 22: return 'NewYork'
        else:             return 'Quiet'
        
    df['Session'] = df['Hour'].apply(get_session)
    
    df['Outcome'] = np.where(df['Profit'] > 0, 'Win', 'Loss')
    
    won = df[df['Profit'] > 0].copy()
    lost = df[df['Profit'] < 0].copy()
    
    print("Enrichment complete.\n")
    return df, won, lost

# ==================== ANALYSIS FUNCTIONS ====================
def show_basic_patterns(df, title):
    print(f"\n=== {title} ===")
    for conf in [f'conf{i}' for i in range(1,11)]:
        if conf in df.columns and not df[conf].isna().all():
            g = df.groupby(conf).agg(
                Trades=('Ticket','count'),
                WinRate=('Outcome', lambda x: (x=='Win').mean()),
                AvgProfit=('Profit','mean')
            )
            print(f"\n{conf}:\n{g.round(3)}")
    
    for col in ['Symbol','Type','DayOfWeek','Hour','Session','Month']:
        if col in df.columns:
            g = df.groupby(col).agg(
                Trades=('Ticket','count'),
                WinRate=('Outcome', lambda x: (x=='Win').mean()),
                AvgProfit=('Profit','mean'),
                TotalProfit=('Profit','sum')
            ).sort_values('AvgProfit', ascending=False)
            print(f"\n{col}:\n{g.round(3)}")

def show_direction_time_patterns(df, title):
    print(f"\n=== {title} - Buy vs Sell by Time ===")
    for col in ['Session','DayOfWeek','Month']:
        if col in df.columns:
            g = df.groupby(['Type', col]).agg(
                Trades=('Ticket','count'),
                WinRate=('Outcome', lambda x: (x=='Win').mean()),
                AvgProfit=('Profit','mean'),
                TotalProfit=('Profit','sum')
            ).round(3)
            print(f"\nBy {col}:\n{g}")

def show_per_symbol_time_patterns(df, title):
    print(f"\n=== {title} - Per Symbol by Time ===")
    for col in ['DayOfWeek','Session','Month']:
        if col in df.columns:
            g = df.groupby(['Symbol', col]).agg(
                Trades=('Ticket','count'),
                WinRate=('Outcome', lambda x: (x=='Win').mean()),
                AvgProfit=('Profit','mean')
            ).round(3)
            print(f"\nBy {col}:\n{g.sort_values(['Symbol','AvgProfit'], ascending=[True,False])}")

def show_deep_patterns(df, title):
    print(f"\n=== {title} Deep Metrics ===")
    cols = ['atr','vol_elevation','Profit','Size','Duration','Hour']
    available = [c for c in cols if c in df.columns]
    if available:
        print(df[available].describe().round(2))
    
    features = df[['atr','vol_elevation','Duration','Profit']].dropna()
    if len(features) > 10:
        print("Performing clustering...")
        scaled = StandardScaler().fit_transform(features)
        df.loc[features.index,'Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
        print("\nClusters:\n", df.groupby('Cluster').agg({
            'Ticket':'count',
            'atr':'mean',
            'vol_elevation':'mean',
            'Duration':'mean',
            'Profit':'mean'
        }).round(2))
    else:
        print("Insufficient data for clustering.")

def generate_recommendations(full, won, lost):
    print("\n" + "="*70)
    print("ACTIONABLE RECOMMENDATIONS")
    print("="*70)
    wr = (full['Profit']>0).mean()
    print(f"Overall Win Rate: {wr:.1%} ({len(won)} wins / {len(lost)} losses)")
    
    sym = full.groupby('Symbol').agg(
        Trades=('Ticket','count'),
        WinRate=('Outcome',lambda x:(x=='Win').mean()),
        AvgProfit=('Profit','mean')
    )
    bad_sym = sym[(sym['WinRate']<0.4) | (sym['AvgProfit']<0)]
    if not bad_sym.empty:
        print("\n→ Disable/restrict these symbols:\n", bad_sym.round(3))
    
    for tcol in ['DayOfWeek','Session']:
        g = full.groupby(['Type',tcol]).agg(
            WinRate=('Outcome',lambda x:(x=='Win').mean()),
            AvgProfit=('Profit','mean')
        )
        bad = g[(g['WinRate']<0.4) | (g['AvgProfit']<0)]
        if not bad.empty:
            print(f"\n→ Avoid {tcol} for these directions:\n", bad.round(3))
    
    print("\n→ Check individual confluences above: disable any where confX=1 hurts win rate/profit.")
    if 'vol_elevation' in full.columns and not lost.empty and lost['vol_elevation'].mean() > 50:
        print("→ Tighten daily range volatility filter (current losses in high-vol days)")

# ==================== MAIN ====================
print("Starting MultiPair_3d Trade Analyzer\n")

# Load deals with robust fallbacks
df_deals = load_deals_csv(trades_csv)
print(f"Raw deals rows: {len(df_deals)}")

df_deals['Time'] = pd.to_datetime(df_deals['Time'], errors='coerce')
df_deals['Profit'] = pd.to_numeric(df_deals['Profit'], errors='coerce').fillna(0)
df_deals['Size'] = pd.to_numeric(df_deals['Size'], errors='coerce')
df_deals['Ticket'] = df_deals['Ticket'].astype(str)

df_deals = df_deals.sort_values(['Ticket', 'Time']).reset_index(drop=True)

trades_list = []
skipped = 0
for ticket, group in df_deals.groupby('Ticket'):
    if len(group) < 2:
        skipped += 1
        continue
    open_time = group['Time'].min()
    close_time = group['Time'].max()
    profit = group['Profit'].sum()
    if abs(profit) < 0.01:
        skipped += 1
        continue
    entry_row = group.iloc[0]
    trades_list.append({
        'Time': open_time,
        'Close Time': close_time,
        'Ticket': ticket,
        'Symbol': entry_row['Symbol'],
        'Type': entry_row['Type'],
        'Size': entry_row['Size'],
        'Profit': profit
    })

df = pd.DataFrame(trades_list)
if len(df) == 0:
    print("No complete trades found. Check your backtest export.")
    exit()

df = df.sort_values('Time').reset_index(drop=True)
df['Ticket'] = df['Ticket'].astype(str)
print(f"Processed {len(df)} complete trades (skipped {skipped}).\n")

# Parse log and enrich
decisions = parse_log_file(log_csv)
full, won, lost = load_data(df, decisions)

# Run analyses
show_basic_patterns(full, "Overall Trades")
show_basic_patterns(won, "Winning Trades")
show_basic_patterns(lost, "Losing Trades")
show_direction_time_patterns(full, "Overall")
show_per_symbol_time_patterns(full, "Overall")
show_deep_patterns(full, "Overall")
generate_recommendations(full, won, lost)

print("\nAnalysis complete!")
  
