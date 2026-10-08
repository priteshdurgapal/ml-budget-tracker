"""
Parser for Standard Chartered Bank India savings account statements from PDF text.

Aligned with the user's existing ICICI parser style:
- parse_*_text(text) returns list[dict]
- parse_*_statement(pdf_path) returns pandas DataFrame
- boilerplate stripping is separated from transaction parsing
- parsing is anchor-based on lines that begin with transaction date + value date

Layout observed in sample Standard Chartered statements:
Date | Value Date | Description | Cheque | Deposit | Withdrawal | Balance

Transaction anchors start with two full dates on one physical line and end with
amount fields on the same line or on following wrapped lines. We collect wrapped
lines until we encounter the tail amount+balance pair, then infer debit/credit
from description keywords and table semantics.
"""

import re
import pandas as pd
import pdfplumber

DATE_PREFIX_LINE = re.compile(
    r'^(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\s+(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\s+(.*)$',
    re.IGNORECASE,
)
DATE_SPLIT_LINE = re.compile(r'^(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})$', re.IGNORECASE)
AMOUNT_RE = re.compile(r'^[\d,]+\.\d{2}$')
TABLE_START_PATTERN = re.compile(r'Date\s+Value\s+Date\s+Description\s+Cheque\s+Deposit\s+Withdrawal\s+Balance', re.IGNORECASE)
END_MARKERS = [r'^Total\s+', r'^REWARD POINTS STATEMENT', r'^Dear Client,']
BOILERPLATE_PATTERNS = [
    r'MR PRITESH DURGAPAL Page \d+ of \d+',
    r'Date\s+Value\s+Date\s+Description\s+Cheque\s+Deposit\s+Withdrawal\s+Balance',
]

MONTHS = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
    'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
}


def _to_iso(date_str: str) -> str:
    day, mon, year = date_str.split()
    return f'{int(year):04d}-{MONTHS[mon.lower()]}-{int(day):02d}'


def _clean_boilerplate(text: str) -> str:
    start_m = TABLE_START_PATTERN.search(text)
    if start_m:
        text = text[start_m.end():]

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.MULTILINE)

    earliest = None
    for marker in END_MARKERS:
        m = re.search(marker, text, re.IGNORECASE | re.MULTILINE)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()
    if earliest is not None:
        text = text[:earliest]

    return text


def _combine_split_dates(lines: list[str]) -> list[str]:
    out = []
    i = 0
    while i < len(lines):
        a = lines[i].strip()
        b = lines[i + 1].strip() if i + 1 < len(lines) else None
        if b and re.fullmatch(r'\d{1,2}\s+[A-Za-z]{3}', a, re.IGNORECASE) and re.fullmatch(r'\d{4}', b):
            out.append(f'{a} {b}')
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def parse_scb_savings_text(text: str) -> list[dict]:
    opening_balance = None
    opening_m = re.search(r'BALANCE FORWARD\s+([\d,]+\.\d{2})', text, re.IGNORECASE)
    if opening_m:
        opening_balance = float(opening_m.group(1).replace(',', ''))

    text = _clean_boilerplate(text)
    lines = [re.sub(r'\s+', ' ', l).strip() for l in text.split('\n') if l.strip()]
    lines = _combine_split_dates(lines)

    records = []
    prev_balance = opening_balance
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.upper().startswith('BALANCE FORWARD'):
            i += 1
            continue
        if any(re.match(marker, line, re.IGNORECASE) for marker in END_MARKERS):
            break

        m = DATE_PREFIX_LINE.match(line)
        if not m:
            i += 1
            continue

        txn_date, value_date, remainder = m.groups()
        parts = [remainder]
        block = [line]
        j = i + 1
        description = None
        amount = None
        balance = None

        while True:
            merged = ' '.join(parts).strip()
            tokens = merged.split()
            if len(tokens) >= 2 and AMOUNT_RE.match(tokens[-1]) and AMOUNT_RE.match(tokens[-2]):
                amount = float(tokens[-2].replace(',', ''))
                balance = float(tokens[-1].replace(',', ''))
                description = ' '.join(tokens[:-2]).strip()
                break
            if j >= len(lines):
                break
            if DATE_PREFIX_LINE.match(lines[j]):
                break
            if any(re.match(marker, lines[j], re.IGNORECASE) for marker in END_MARKERS):
                break
            parts.append(lines[j])
            block.append(lines[j])
            j += 1

        if description and amount is not None and balance is not None:
            if prev_balance is None:
                txn_type = 'unknown'
                deposit = amount if any(k in description.lower() for k in ['interest', 'salary', 'credit']) else 0.0
                withdrawal = 0.0 if deposit else amount
            else:
                desc = description.lower()
                credit_keywords = ['salary', 'interest', 'credit', 'neft', 'imps', 'rtgs', 'deposit']
                debit_keywords = ['withdrawal', 'charges', 'fee', 'igst', 'atm', 'debit']
                if any(k in desc for k in credit_keywords) and not any(k in desc for k in debit_keywords):
                    txn_type = 'credit'
                elif any(k in desc for k in debit_keywords) and not any(k in desc for k in credit_keywords):
                    txn_type = 'debit'
                else:
                    txn_type = 'credit' if balance > prev_balance else 'debit'
                deposit = amount if txn_type == 'credit' else 0.0
                withdrawal = amount if txn_type == 'debit' else 0.0

            records.append({
                'date': _to_iso(txn_date),
                'value_date': _to_iso(value_date),
                'description': description,
                'amount': amount,
                'deposit': deposit,
                'withdrawal': withdrawal,
                'balance': balance,
                'txn_type': txn_type,
            })
            prev_balance = balance
            i = j
            continue

        i += 1

    return records


def parse_scb_savings_statement(pdf_path: str) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        full_text = '\n'.join(p.extract_text() for p in pdf.pages if p.extract_text())

    records = parse_scb_savings_text(full_text)
    df = pd.DataFrame(records)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value_date'] = pd.to_datetime(df['value_date'], errors='coerce')
        df['source'] = 'scb_savings'
    return df


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python scb_savings_parser_aligned.py <path_to_statement.pdf>')
        raise SystemExit(1)

    with pdfplumber.open(sys.argv[1]) as pdf:
        num_pages = len(pdf.pages)

    df = parse_scb_savings_statement(sys.argv[1])
    print(df.to_string())
    print(f'\nPages in PDF: {num_pages}')
    print(f'Total transactions parsed: {len(df)}')
    if 'txn_type' in df.columns and (df['txn_type'] == 'unknown').any():
        print(f"NOTE: {(df['txn_type'] == 'unknown').sum()} row(s) have unknown debit/credit type.")