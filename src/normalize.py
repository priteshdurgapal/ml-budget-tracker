"""
Normalizes each bank parser's raw output into one unified transaction schema:

    date | description | amount | txn_type | bank | account_type |
    merchant_category | reward_points | source_file

txn_type is always 'debit' or 'credit' (or 'opening_balance'/'unknown' for
the savings parser's edge cases - these get filtered out downstream, not
treated as real transactions).

account_type is either 'credit_card' or 'savings' - this matters later for
transfer-detection, since a credit card bill payment shows up as a CREDIT
on the card and a DEBIT on the savings account, and we need to know which
account each row came from to match those pairs.
"""

import pandas as pd

UNIFIED_COLUMNS = [
    'date', 'description', 'amount', 'txn_type',
    'bank', 'account_type', 'merchant_category', 'reward_points', 'source_file'
]


def normalize(df: pd.DataFrame, bank: str, account_type: str, source_file: str) -> pd.DataFrame:
    """Map a single parser's output DataFrame into the unified schema."""
    if df.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    out = pd.DataFrame()
    out['date'] = df['date']
    out['description'] = df['description']
    out['amount'] = df['amount']
    out['txn_type'] = df['txn_type']
    out['bank'] = bank
    out['account_type'] = account_type

    # merchant_category only exists for YES Bank - default to None otherwise
    out['merchant_category'] = df['merchant_category'] if 'merchant_category' in df.columns else None

    # reward points column name differs slightly across parsers (rewards_points
    # vs reward_points) - handle both, default to None if absent (e.g. savings)
    if 'reward_points' in df.columns:
        out['reward_points'] = df['reward_points']
    elif 'rewards_points' in df.columns:
        out['reward_points'] = df['rewards_points']
    else:
        out['reward_points'] = None

    out['source_file'] = source_file

    return out[UNIFIED_COLUMNS]