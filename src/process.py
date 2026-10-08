"""
Main entry point for the expense analyzer.

Run this after dropping PDF statements into their respective folders under
data/raw_pdfs/<bank_folder>/. It parses every PDF in each folder, normalizes
the output to a common schema, and writes one combined CSV.

Usage:
    python src/run_all.py
"""

import glob
import os
import sys
import traceback

import pandas as pd

# Make sibling modules importable regardless of what directory this is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalize import normalize
from parsers.hd_parser import parse_hdfc_cc_statement
from parsers.ic_parser import parse_icici_cc_statement
from parsers.ys_parser import parse_yesbank_cc_statement
from parsers.ic_sav_parser import parse_icici_savings_regular_statement
from parsers.sc_parser import parse_scb_savings_statement
from parsers.amex_parser import parse_amex_statement


# Registry: folder path -> (parser function, bank name, account type)
# Add a new row here whenever a new bank/account is parsed.
PARSER_REGISTRY = [
    {
        'folder': 'data/hdfc',
        'parser': parse_hdfc_cc_statement,
        'bank': 'hdfc',
        'account_type': 'credit_card',
    },
    {
        'folder': 'data/icici_credit',
        'parser': parse_icici_cc_statement,
        'bank': 'icici',
        'account_type': 'credit_card',
    },
    {
        'folder': 'data/yes',
        'parser': parse_yesbank_cc_statement,
        'bank': 'yesbank',
        'account_type': 'credit_card',
    },
    {
        'folder': 'data/icici_savings',
        'parser': parse_icici_savings_regular_statement,
        'bank': 'icici',
        'account_type': 'savings',
    },
    {
        'folder': 'data/scb',
        'parser': parse_scb_savings_statement,
        'bank': 'scb',
        'account_type': 'savings',
    },
    {
        'folder': 'data/amex',
        'parser': parse_amex_statement,
        'bank': 'amex',
        'account_type': 'credit_card',
    },    
]

OUTPUT_PATH = 'data/processed/all_transactions.csv'


def process_folder(entry: dict) -> pd.DataFrame:
    """Parse every PDF in one bank's folder, normalize, and combine."""
    folder = entry['folder']
    parser = entry['parser']
    bank = entry['bank']
    account_type = entry['account_type']

    pdf_paths = sorted(glob.glob(os.path.join(folder, '*.pdf')))

    if not pdf_paths:
        print(f"  [SKIP] No PDFs found in {folder}")
        return pd.DataFrame()

    all_normalized = []

    for pdf_path in pdf_paths:
        try:
            raw_df = parser(pdf_path)
            n_rows = len(raw_df)

            if n_rows == 0:
                print(f"  [WARNING] {pdf_path} -> 0 transactions parsed. "
                      f"This is suspicious - check the file manually.")
                continue

            normalized_df = normalize(raw_df, bank=bank, account_type=account_type,
                                       source_file=os.path.basename(pdf_path))
            all_normalized.append(normalized_df)
            print(f"  [OK] {pdf_path} -> {n_rows} transactions")

        except Exception as e:
            # Fail loudly per-file, but don't let one bad statement stop
            # the whole run - other months/banks should still process.
            print(f"  [FAILED] {pdf_path}")
            print(f"           {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)

    if not all_normalized:
        return pd.DataFrame()

    return pd.concat(all_normalized, ignore_index=True)


def main():
    print("=" * 70)
    print("EXPENSE ANALYZER - Processing all bank statements")
    print("=" * 70)

    all_banks_data = []

    for entry in PARSER_REGISTRY:
        print(f"\nProcessing: {entry['folder']} ({entry['bank']} / {entry['account_type']})")
        df = process_folder(entry)
        if not df.empty:
            all_banks_data.append(df)

    if not all_banks_data:
        print("\nNo transactions parsed from any source. Nothing to write.")
        return

    combined = pd.concat(all_banks_data, ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print(f"DONE: {len(combined)} total transactions written to {OUTPUT_PATH}")
    print("=" * 70)

    # Quick sanity summary - always worth a glance before trusting the output
    print("\nBreakdown by bank / account type:")
    print(combined.groupby(['bank', 'account_type']).size().to_string())

    print("\nDate range:", combined['date'].min(), "to", combined['date'].max())

    n_unusual_types = combined[~combined['txn_type'].isin(['debit', 'credit'])].shape[0]
    if n_unusual_types > 0:
        print(f"\nNOTE: {n_unusual_types} rows have txn_type outside debit/credit "
              f"(e.g. 'opening_balance', 'unknown') - review these before "
              f"including them in expense totals.")


if __name__ == "__main__":
    main()