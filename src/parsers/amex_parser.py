"""
Parser for American Express India credit card statements from PDF text.

Aligned with the user's existing ICICI parser style:
- parse_*_text(text) returns list[dict]
- parse_*_statement(pdf_path) returns pandas DataFrame
- boilerplate stripping is separated from transaction parsing
- parsing is anchor-based on dated transaction lines

Layout observed in sample Amex statements:
Details | Foreign Spending | Amount Rs

Transaction lines begin with a month-name date and end with one decimal amount.
A standalone line 'CR' may appear after a transaction or card-number line; in the
sample statement, it marks the NEXT transaction as a credit. Credits can also be
inferred from description keywords such as 'PAYMENT RECEIVED'.
"""

import re
import pandas as pd
import pdfplumber


MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
}

MONTH_NUM_TO_PADDED = {
    1: '01', 2: '02', 3: '03', 4: '04', 5: '05', 6: '06',
    7: '07', 8: '08', 9: '09', 10: '10', 11: '11', 12: '12',
}

DATE_AMOUNT_LINE = re.compile(
    r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
    r'(\d{1,2})\s+(.*?)\s+([\d,]+\.\d{2})\s*$',
    re.IGNORECASE,
)
STATEMENT_DATE_RE = re.compile(r'\b(\d{2})/(\d{2})/(\d{2}|\d{4})\b')
TABLE_START_PATTERN = re.compile(r'Details\s+Foreign Spending\s+Amount Rs', re.IGNORECASE)
END_MARKERS = [
    r'New domestic transactions for',
    r'Total of other account transactions',
    r'Payment Information',
    r'CardMember Offers and Information',
    r'We have made changes',
]
BOILERPLATE_PATTERNS = [
    r'American Express.*?Statement of Account',
    r'Prepared for.*',
    r'Card Number\s+XXXX-XXXXXX-\d+\s*CR?',
    r'Page \d+ of \d+',
    r'Details\s+Foreign Spending\s+Amount Rs',
    r'OTHER ACCOUNT TRANSACTIONS',
]
CREDIT_KEYWORDS = [r'PAYMENT RECEIVED', r'REFUND', r'REVERSAL', r'CASHBACK']


def _clean_boilerplate(text: str) -> str:
    earliest_start = None
    for m in TABLE_START_PATTERN.finditer(text):
        if earliest_start is None or m.start() < earliest_start:
            earliest_start = m.start()
    if earliest_start is not None:
        text = text[earliest_start:]

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    earliest_end = None
    for marker in END_MARKERS:
        m = re.search(marker, text, re.IGNORECASE)
        if m and (earliest_end is None or m.start() < earliest_end):
            earliest_end = m.start()
    if earliest_end is not None:
        text = text[:earliest_end]

    return text


def _parse_statement_date(text: str):
    m = STATEMENT_DATE_RE.search(text)
    if not m:
        return None, None, None
    day, month, year = m.groups()
    year = int(year)
    if year < 100:
        year += 2000
    return int(day), int(month), year


def _infer_txn_year(txn_month_num: int, statement_month: int, statement_year: int) -> int:
    if txn_month_num > statement_month:
        return statement_year - 1
    return statement_year


def parse_amex_statement_text(text: str) -> list[dict]:
    _, statement_month, statement_year = _parse_statement_date(text)
    text = _clean_boilerplate(text)
    lines = [re.sub(r'\s+', ' ', l).strip() for l in text.split('\n') if l.strip()]

    records = []

    for line in lines:
        if line.upper() == 'CR':
            # 'CR' marks the PREVIOUS parsed transaction as a credit,
            # not the next one - it appears directly after that row's
            # amount in the raw PDF text.
            if records:
                records[-1]['txn_type'] = 'credit'
            continue

        m = DATE_AMOUNT_LINE.match(line)
        if not m:
            continue

        month_name, day_str, description, amount_str = m.groups()
        txn_month_num = MONTHS[month_name.lower()]
        day = int(day_str)

        if statement_year is not None and statement_month is not None:
            txn_year = _infer_txn_year(txn_month_num, statement_month, statement_year)
        else:
            txn_year = 1900

        date_str = f"{txn_year:04d}-{MONTH_NUM_TO_PADDED[txn_month_num]}-{day:02d}"
        amount = float(amount_str.replace(',', ''))

        desc_upper = description.upper()
        txn_type = 'credit' if any(k in desc_upper for k in CREDIT_KEYWORDS) else 'debit'

        records.append({
            'date': date_str,
            'description': description.strip(),
            'amount': amount,
            'balance': None,
            'txn_type': txn_type,
        })

    return records


def parse_amex_statement(pdf_path: str) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        full_text = '\n'.join(p.extract_text() for p in pdf.pages if p.extract_text())

    records = parse_amex_statement_text(full_text)
    df = pd.DataFrame(records)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['source'] = 'amex_statement'
    return df


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python amex_statement_parser_aligned.py <path_to_statement.pdf>')
        raise SystemExit(1)

    with pdfplumber.open(sys.argv[1]) as pdf:
        num_pages = len(pdf.pages)

    df = parse_amex_statement(sys.argv[1])
    print(df.to_string())
    print(f'\nPages in PDF: {num_pages}')
    print(f'Total transactions parsed: {len(df)}')
    if 'txn_type' in df.columns and (df['txn_type'] == 'unknown').any():
        print(f"NOTE: {(df['txn_type'] == 'unknown').sum()} row(s) have unknown debit/credit type.")