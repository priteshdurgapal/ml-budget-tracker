"""
Local ML + rules pipeline for normalized bank/card transaction CSV files.

Input CSV columns expected:
- date
- description
- amount
- txn_type
- bank
- account_type
- merchant_category
- reward_points
- source_file

What this script does:
1. Cleans and normalizes transaction descriptions.
2. Detects likely internal money movement / self-transfers across banks.
3. Auto-assigns merchant categories using rules + optional ML from existing labels.
4. Adds date features (day, weekday, week, month).
5. Produces monthly spend summaries and a simple monthly spend forecast.
6. Saves enriched CSV plus summary CSV files and monthly graphs.

Usage:
    python transaction_ml_pipeline.py input.csv --outdir output_dir
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


EXPLICIT_TRANSFER_PATTERNS = [
    r'bil/neft', r'bil/inft', r'bil/onl', r'\bneft\b', r'\brtgs\b', r'imps p2a',
    r'cc billpay', r'billpay-', r'fund transfer', r'to self', r'from self',
    r'own account', r'pritesh du', r'payment received', r'infinity payment received',
    r'neft credit card payment', r'p2a charges',
]
INVESTMENT_PATTERNS = [
    r'installment for deal', r'trf to new deposit', r'groww', r'zerodha', r'axismf',
    r'canrob\.bdpg', r'canara rob', r'\bppf\b', r'axis asset', r'mutual fund',
    r'\bsip\b', r'coin\.', r'upstox', r'nps\b', r'edgro fina', r'aditya birla', r'birla sunlife',
    r'dityabirl', r'absl',
]
SCB_DEAL_REF = re.compile(r'^\d{4}-\d{12}$')
MERCHANT_PAYMENT_KEYWORDS = [
    'zomato', 'swiggy', 'blinkit', 'instamart', 'bigbasket', 'zepto', 'eatsure',
    'amazon', 'flipkart', 'myntra', 'westside', 'nykaa', 'ikea', 'jockey',
    'airtel', 'jio', 'google ind', 'gpay', 'paytmqr', 'paytm-', 'payu', 'pyu',
    'razorpay', 'policybazaar', 'cloudnine', 'maxlife', 'elior indi',
    'restaurant', 'cafe', 'pvr', 'inox', 'uber', 'ola', 'rapido', 'bescom',
    'billdesk', 'easebuzz', 'lulu daily', 'kids clini', 'ramdev med', 'bagmane',
    'halli thot', 'moocha', 'noodle bar', 'titan worl', 'apollo', 'birla institute',
]
NON_EXPENSE_CATEGORIES = {
    'internal_transfer', 'investment', 'salary_income', 'interest_income',
    'refund_reversal', 'atm_cash',
}

CATEGORY_RULES = {
    'food_delivery': [r'zomato', r'swiggy', r'eatsure', r'zepto cafe'],
    'groceries': [r'blinkit', r'zepto', r'instamart', r'bigbasket', r'bbnow', r'grocery'],
    'dining': [r'restaurant', r'cafe', r'coffee', r'barbeque', r'bistro', r'eatfit'],
    'transport': [r'uber', r'ola', r'rapido', r'irctc', r'metro', r'fuel', r'petrol', r'diesel'],
    'shopping': [r'amazon', r'flipkart', r'myntra', r'ajio', r'westside', r'nykaa'],
    'utilities': [r'electricity', r'water', r'gas', r'billdesk', r'broadband', r'jio', r'airtel', r'bescom'],
    'fees_charges': [r'fee', r'charge', r'gst', r'igst', r'finance charge', r'late payment'],
    'salary_income': [r'salary', r'payroll', r'paytm payroll'],
    'interest_income': [r'interest'],
    'refund_reversal': [r'refund', r'reversal', r'cashback'],
    'atm_cash': [r'atm withdrawal', r'cash withdrawal', r'nfs/cash'],
    'travel': [r'mmt', r'makemytrip', r'yatra', r'air india', r'indigo', r'akasa', r'vistara', r'booking\.com'],
    'insurance': [r'insurance', r'policybazaar', r'hdfc ergo', r'icici lombard'],
    'investment': [
        r'zerodha', r'groww', r'coin', r'mutual fund', r'sip', r'nps', r'upstox',
        r'installment for deal', r'trf to new deposit', r'axismf', r'canrob\.bdpg',
    ],
    'emi_loan': [r'emi', r'loan', r'bajaj finance'],
    'rent': [r'rent'],
}

GATEWAY_PREFIXES = [
    r'^payu\*', r'^pyu\*', r'^razorpay\*', r'^billdesk\*', r'^juspay\*', r'^ccavenue\*',
    r'^amazon pay\*', r'^paytm\*', r'^phonepe\*', r'^gpay\*', r'^airpay\*'
]


def normalize_description(text: str) -> str:
    if pd.isna(text):
        return ''
    x = str(text).lower().strip()
    x = x.replace('\n', ' ')
    x = re.sub(r'[^a-z0-9\s\*\-/]', ' ', x)
    for patt in GATEWAY_PREFIXES:
        x = re.sub(patt, '', x)
    x = re.sub(r'\b\d{6,}\b', ' ', x)
    x = re.sub(r'\b[a-z]{2,}\d{6,}[a-z0-9]*\b', ' ', x)
    x = re.sub(r'\b(?:gurugram|bangalore|bengaluru|mumbai|delhi|faridabad|hyderabad|pune|chennai)\b', ' ', x)
    x = re.sub(r'\b(?:ltd|pvt|private|bank|india|road|main)\b', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x


def canonical_merchant(clean_desc: str) -> str:
    tokens = clean_desc.split()
    if not tokens:
        return 'unknown'
    joined = ' '.join(tokens)
    merchant_map = {
        'zomato': ['zomato'],
        'blinkit': ['blinkit'],
        'swiggy': ['swiggy'],
        'uber': ['uber'],
        'ola': ['ola'],
        'amazon': ['amazon'],
        'flipkart': ['flipkart'],
        'westside': ['westside'],
        'salary': ['salary'],
        'interest': ['interest'],
    }
    for canon, patterns in merchant_map.items():
        if any(p in joined for p in patterns):
            return canon
    return ' '.join(tokens[:3])


def _description_raw(row: pd.Series) -> str:
    return str(row.get('description', '') or '').lower()


def is_p2p_upi_transfer(desc_raw: str) -> bool:
    return bool(re.search(r'upi/[^/]+/[^/]*@', desc_raw))


def is_merchant_payment(row: pd.Series) -> bool:
    desc_raw = _description_raw(row)
    desc_clean = str(row.get('description_clean', '') or '').lower()
    combined = f'{desc_raw} {desc_clean}'
    return any(keyword in combined for keyword in MERCHANT_PAYMENT_KEYWORDS)


def is_investment_movement(row: pd.Series) -> bool:
    desc_raw = _description_raw(row)
    desc_stripped = str(row.get('description', '') or '').strip()
    if SCB_DEAL_REF.match(desc_stripped):
        return True
    return any(re.search(pattern, desc_raw) for pattern in INVESTMENT_PATTERNS)


def detect_self_transfer(row: pd.Series) -> bool:
    desc_raw = _description_raw(row)
    account_type = str(row.get('account_type', '')).lower()

    if is_investment_movement(row):
        return False

    if account_type == 'savings' and is_p2p_upi_transfer(desc_raw):
        return True

    if is_merchant_payment(row):
        return False

    if any(re.search(pattern, desc_raw) for pattern in EXPLICIT_TRANSFER_PATTERNS):
        return True

    if 'self' in desc_raw or 'own account' in desc_raw:
        return True

    if account_type == 'savings':
        if re.search(r'\b(?:neft|rtgs|imps|inft)\b', desc_raw) or desc_raw.startswith('bil/'):
            return True
        if 'upi/' in desc_raw and '@' in desc_raw:
            return True

    return False


def dedupe_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows caused by overlapping statement PDF periods."""
    out = df.copy()
    key_cols = ['date', 'description', 'amount', 'txn_type', 'bank', 'account_type']
    stmt_month = out['source_file'].fillna('').str.extract(r'_(\d{2})_26\.pdf', expand=False)
    out['_stmt_rank'] = pd.to_numeric(stmt_month, errors='coerce').fillna(99)
    out = out.sort_values('_stmt_rank').drop_duplicates(subset=key_cols, keep='first')
    return out.drop(columns=['_stmt_rank']).reset_index(drop=True)


def rule_based_category(row: pd.Series) -> str:
    desc = row.get('description_clean', '')
    if is_investment_movement(row):
        return 'investment'
    if row.get('is_internal_transfer', False):
        return 'internal_transfer'
    if str(row.get('txn_type', '')).lower() == 'credit' and re.search(r'salary', desc):
        return 'salary_income'
    if str(row.get('txn_type', '')).lower() == 'credit' and re.search(r'interest', desc):
        return 'interest_income'
    for category, patterns in CATEGORY_RULES.items():
        for pattern in patterns:
            if re.search(pattern, desc):
                return category
    if str(row.get('txn_type', '')).lower() == 'credit':
        return 'uncategorized_credit'
    return 'uncategorized_expense'


def train_ml_category_model(df: pd.DataFrame) -> Optional[Pipeline]:
    labeled = df[
        df['merchant_category'].notna()
        & (df['merchant_category'].astype(str).str.strip() != '')
        & (~df['merchant_category'].isin(['0', 'unknown', 'uncategorized_expense', 'uncategorized_credit']))
    ].copy()
    if len(labeled) < 30 or labeled['merchant_category'].nunique() < 2:
        return None

    X = (
        labeled['description_clean'].fillna('')
        + ' bank_' + labeled['bank'].fillna('').astype(str)
        + ' acct_' + labeled['account_type'].fillna('').astype(str)
        + ' txn_' + labeled['txn_type'].fillna('').astype(str)
    )
    y = labeled['merchant_category'].astype(str)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])
    model.fit(X, y)
    return model


def assign_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['description_clean'] = df['description'].fillna('').map(normalize_description)
    df['merchant_clean'] = df['description_clean'].map(canonical_merchant)
    df['is_internal_transfer'] = df.apply(detect_self_transfer, axis=1)
    df['category_rule'] = df.apply(rule_based_category, axis=1)

    model = train_ml_category_model(df)
    if model is not None:
        X_all = (
            df['description_clean'].fillna('')
            + ' bank_' + df['bank'].fillna('').astype(str)
            + ' acct_' + df['account_type'].fillna('').astype(str)
            + ' txn_' + df['txn_type'].fillna('').astype(str)
        )
        preds = model.predict(X_all)
        df['category_ml'] = preds
        needs_fill = df['category_rule'].isin(['uncategorized_expense', 'uncategorized_credit'])
        df['merchant_category_final'] = np.where(needs_fill, df['category_ml'], df['category_rule'])
    else:
        df['category_ml'] = None
        df['merchant_category_final'] = df['category_rule']

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
    missing = parsed.isna()
    if missing.any():
        parsed_alt = pd.to_datetime(df.loc[missing, 'date'], errors='coerce', dayfirst=True)
        parsed.loc[missing] = parsed_alt
    df['date'] = parsed
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['month_num'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype('Int64')
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['is_weekend'] = df['day_name'].isin(['Saturday', 'Sunday'])
    return df


def expense_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['amount'] = pd.to_numeric(out['amount'], errors='coerce').fillna(0)
    out = out[out['txn_type'].astype(str).str.lower() == 'debit'].copy()
    out = out[~out['merchant_category_final'].isin(NON_EXPENSE_CATEGORIES)].copy()
    return out


def monthly_spend_summary(df: pd.DataFrame) -> pd.DataFrame:
    exp = expense_view(df)
    return exp.groupby(['month', 'merchant_category_final'], as_index=False)['amount'].sum().sort_values(['month', 'amount'], ascending=[True, False])


def monthly_total_spend(df: pd.DataFrame) -> pd.DataFrame:
    exp = expense_view(df)
    return exp.groupby('month', as_index=False)['amount'].sum().sort_values('month')


def simple_forecast_next_3_months(monthly_df: pd.DataFrame) -> pd.DataFrame:
    m = monthly_df.copy()
    m['month_period'] = pd.PeriodIndex(m['month'], freq='M')
    m = m.sort_values('month_period')
    if len(m) == 0:
        return pd.DataFrame(columns=['month', 'forecast_amount'])

    y = m['amount'].values.astype(float)
    if len(y) == 1:
        forecast_vals = [y[-1]] * 3
    else:
        x = np.arange(len(y))
        coef = np.polyfit(x, y, deg=1)
        future_x = np.arange(len(y), len(y) + 3)
        forecast_vals = np.polyval(coef, future_x)
        forecast_vals = [max(0.0, float(v)) for v in forecast_vals]

    last_period = m['month_period'].iloc[-1]
    future_months = [(last_period + i).strftime('%Y-%m') for i in range(1, 4)]
    return pd.DataFrame({'month': future_months, 'forecast_amount': forecast_vals})


def save_monthly_chart(monthly_df: pd.DataFrame, forecast_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(10, 5))
    if not monthly_df.empty:
        plt.plot(monthly_df['month'], monthly_df['amount'], marker='o', label='Actual spend')
    if not forecast_df.empty:
        full_x = list(monthly_df['month']) + list(forecast_df['month'])
        plt.plot(forecast_df['month'], forecast_df['forecast_amount'], marker='o', linestyle='--', label='Forecast spend')
        plt.xticks(rotation=45)
    else:
        plt.xticks(rotation=45)
    plt.title('Monthly Spend Trend and Forecast')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'monthly_spend_forecast.png', dpi=160)
    plt.close()


def save_day_week_charts(df: pd.DataFrame, outdir: Path) -> None:
    exp = expense_view(df)

    day_df = exp.groupby('day_name', as_index=False)['amount'].sum()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_df['day_name'] = pd.Categorical(day_df['day_name'], categories=order, ordered=True)
    day_df = day_df.sort_values('day_name')
    plt.figure(figsize=(9, 4))
    plt.bar(day_df['day_name'], day_df['amount'])
    plt.title('Spend by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Amount')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outdir / 'spend_by_day_of_week.png', dpi=160)
    plt.close()

    week_df = exp.groupby(['year', 'week'], as_index=False)['amount'].sum()
    if not week_df.empty:
        week_df['year_week'] = week_df['year'].astype(str) + '-W' + week_df['week'].astype(str)
        plt.figure(figsize=(10, 4))
        plt.plot(week_df['year_week'], week_df['amount'], marker='o')
        plt.title('Weekly Spend Trend')
        plt.xlabel('Week')
        plt.ylabel('Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(outdir / 'weekly_spend_trend.png', dpi=160)
        plt.close()


def save_category_spend_chart(df: pd.DataFrame, outdir: Path) -> None:
    """Save a horizontal bar chart of total expense amount by category."""
    exp = expense_view(df)
    category_df = (
        exp.groupby('merchant_category_final', as_index=False)['amount']
        .sum()
        .sort_values('amount', ascending=True)
    )

    if category_df.empty:
        return

    chart_height = max(5, 0.45 * len(category_df))
    fig, ax = plt.subplots(figsize=(11, chart_height))
    bars = ax.barh(category_df['merchant_category_final'], category_df['amount'], color='#4C78A8')

    ax.set_title('Total Spending by Category')
    ax.set_xlabel('Amount (INR)')
    ax.set_ylabel('Category')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'₹{value:,.0f}'))
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    max_amount = category_df['amount'].max()
    for bar, amount in zip(bars, category_df['amount']):
        ax.text(
            bar.get_width() + max_amount * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'₹{amount:,.0f}',
            va='center',
            fontsize=9,
        )

    ax.set_xlim(0, max_amount * 1.18)
    fig.tight_layout()
    fig.savefig(outdir / 'total_spend_by_category.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Normalize, categorize, and forecast bank/card transaction CSV data.')
    ap.add_argument('input_csv', help='Normalized input CSV')
    ap.add_argument('--outdir', default='ml_output', help='Output directory')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = dedupe_transactions(df)
    df = assign_categories(df)
    df = add_date_features(df)

    monthly_cat = monthly_spend_summary(df)
    monthly_total = monthly_total_spend(df)
    forecast_df = simple_forecast_next_3_months(monthly_total)

    df.to_csv(outdir / 'transactions_enriched.csv', index=False)
    monthly_cat.to_csv(outdir / 'monthly_spend_by_category.csv', index=False)
    monthly_total.to_csv(outdir / 'monthly_total_spend.csv', index=False)
    forecast_df.to_csv(outdir / 'monthly_spend_forecast.csv', index=False)

    save_monthly_chart(monthly_total, forecast_df, outdir)
    save_day_week_charts(df, outdir)
    save_category_spend_chart(df, outdir)

    print(f'Enriched transactions saved to: {outdir / "transactions_enriched.csv"}')
    print(f'Monthly category summary saved to: {outdir / "monthly_spend_by_category.csv"}')
    print(f'Monthly total spend saved to: {outdir / "monthly_total_spend.csv"}')
    print(f'Forecast saved to: {outdir / "monthly_spend_forecast.csv"}')
    print('Charts saved: monthly_spend_forecast.png, spend_by_day_of_week.png, '
          'weekly_spend_trend.png, total_spend_by_category.png')


if __name__ == '__main__':
    main()
