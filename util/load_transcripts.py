import os
import re
import pandas as pd
from datetime import datetime

DATA_DIR = '../data/transcripts'

def extract_metadata(filename, ticker):
    try:
        date_str = filename.replace(f'-{ticker}.txt', '')
        date = datetime.strptime(date_str, "%Y-%b-%d")
        year = date.year
        month = date.month
        quarter = f"Q{((month - 1) // 3) + 1}"
        return quarter, year, date
    except Exception:
        return None, None, None

def load_all_transcripts(data_dir=DATA_DIR):
    rows = []

    for ticker in os.listdir(data_dir):
        company_dir = os.path.join(data_dir, ticker)
        if not os.path.isdir(company_dir):
            continue

        for fname in os.listdir(company_dir):
            if not fname.endswith('.txt'):
                continue

            filepath = os.path.join(company_dir, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            quarter, year, date = extract_metadata(fname, ticker)
            rows.append({
                'ticker': ticker,
                'quarter': quarter,
                'date': date,
                'text': text
            })

    df = pd.DataFrame(rows)
    return df
