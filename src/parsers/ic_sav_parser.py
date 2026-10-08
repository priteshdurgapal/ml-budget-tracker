"""
Parser for ICICI Bank Savings Account statements - "regular statement" format
(as opposed to the OpTransactionHistory export format - different template,
same bank).

Layout: DATE | MODE | PARTICULARS | DEPOSITS | WITHDRAWALS | BALANCE

Real extracted text shows the DATE+AMOUNT+BALANCE (and optionally a short
MODE keyword like "OTHER ATMS" squished directly after the date with no
space) always appear together, tightly packed, on ONE physical text line -
even when the surrounding PARTICULARS/description wraps across several
other lines before and after that line. So we anchor on "a line that starts
with a date and ends with decimal amount(s)" rather than trying to delimit
by date position alone.

The very first row is always "B/F" (balance brought forward) with only a
balance, no amount - this lets us reliably determine debit/credit via
running-balance comparison from row 1 onward (unlike the OpTransactionHistory
format, which has no such opening-balance reference).
"""

import re
import pandas as pd
import pdfplumber

# A full line that starts with a hyphenated date and ends with TWO decimal
# amounts (deposit-or-withdrawal amount, then balance).
TWO_AMOUNT_LINE = re.compile(
    r'^(\d{2}-\d{2}-\d{4})\s*(.*?)\s*([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$'
)
# A full line that starts with a hyphenated date and ends with ONE decimal
# amount only (the B/F opening-balance row).
ONE_AMOUNT_LINE = re.compile(
    r'^(\d{2}-\d{2}-\d{4})\s*(.*?)\s*([\d,]+\.\d{2})\s*$'
)

# Recurring page boilerplate to strip before parsing
BOILERPLATE_PATTERNS = [
    r'Statement of Transactions in Savings Account \S+ in INR for the period[^\n]*',
    r'DATE\s+MODE\s+PARTICULARS\s+DEPOSITS\s+WITHDRAWALS\s+BALANCE',
    r'Total:\s*[\d,]+\.\d{2}\s+[\d,]+\.\d{2}\s+[\d,]+\.\d{2}',
    r'Page \d+ of \d+ M-[\w-]+',
]
END_MARKERS = [r'Account Related Other Information', r'Sincerely,']


TABLE_START_PATTERN = re.compile(
    r'DATE\s+MODE\s+PARTICULARS\s+DEPOSITS\s+WITHDRAWALS\s+BALANCE',
    re.IGNORECASE
)


def _clean_boilerplate(text: str) -> str:
    # First, cut everything before the FIRST occurrence of the real table
    # header - this drops the address block, statement summary, and
    # recurring-deposit table that precede the transaction table on page 1.
    start_m = TABLE_START_PATTERN.search(text)
    if start_m:
        text = text[start_m.end():]

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    earliest = None
    for marker in END_MARKERS:
        m = re.search(marker, text, re.IGNORECASE)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()
    if earliest is not None:
        text = text[:earliest]

    return text


def _find_anchors(lines: list[str]):
    """Find every line that is a transaction anchor (date + amount(s))."""
    anchors = []
    for idx, line in enumerate(lines):
        m2 = TWO_AMOUNT_LINE.match(line.strip())
        if m2:
            date_str, mid_text, amount_str, balance_str = m2.groups()
            anchors.append({
                'line_idx': idx,
                'date': date_str,
                'mid_text': mid_text,
                'amount': float(amount_str.replace(',', '')),
                'balance': float(balance_str.replace(',', '')),
                'is_opening': False,
            })
            continue
        m1 = ONE_AMOUNT_LINE.match(line.strip())
        if m1:
            date_str, mid_text, balance_str = m1.groups()
            anchors.append({
                'line_idx': idx,
                'date': date_str,
                'mid_text': mid_text,
                'amount': None,
                'balance': float(balance_str.replace(',', '')),
                'is_opening': True,
            })
    return anchors


# A line is a "continuation fragment" (the tail of a reference number that
# wrapped onto its own line) if it contains a long digit run, or is itself
# just a short leftover hash tail like "9/" or "3/". Anything else (a
# merchant/payer name, a UPI/BIL reference string) starts a NEW transaction.
_LONG_DIGIT_RUN = re.compile(r'\d{6,}')
_SHORT_HEX_TAIL = re.compile(r'^[0-9a-fA-F]{1,6}/?$')


def _is_continuation_fragment(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _LONG_DIGIT_RUN.search(stripped):
        return True
    if _SHORT_HEX_TAIL.match(stripped):
        return True
    return False


def parse_icici_savings_regular_text(text: str) -> list[dict]:
    text = _clean_boilerplate(text)
    lines = [l for l in text.split('\n')]

    anchors = _find_anchors(lines)
    n = len(anchors)
    if n == 0:
        return []

    extra_before = [''] * n  # preamble text (before the anchor line) per row
    extra_after = [''] * n   # continuation text (after the anchor line) per row

    def _split_gap(gap_lines):
        """Return (continuation_part, preamble_part) - continuation_part
        belongs to the PREVIOUS row, preamble_part belongs to the NEXT row."""
        split_idx = len(gap_lines)
        for idx, line in enumerate(gap_lines):
            if not _is_continuation_fragment(line):
                split_idx = idx
                break
        return gap_lines[:split_idx], gap_lines[split_idx:]

    # gap before the very first anchor
    pre_lines = [l.strip() for l in lines[:anchors[0]['line_idx']] if l.strip()]
    _, preamble0 = _split_gap(pre_lines)
    extra_before[0] = ' '.join(preamble0)

    # gaps between consecutive anchors
    for i in range(n - 1):
        gap_lines = [l.strip() for l in lines[anchors[i]['line_idx'] + 1: anchors[i + 1]['line_idx']] if l.strip()]
        continuation_part, preamble_part = _split_gap(gap_lines)
        extra_after[i] = ' '.join(continuation_part)
        extra_before[i + 1] = ' '.join(preamble_part)

    # gap after the very last anchor - keep only the continuation part
    # (leftover text after that is very likely trailing page boilerplate)
    post_lines = [l.strip() for l in lines[anchors[-1]['line_idx'] + 1:] if l.strip()]
    continuation_last, _ = _split_gap(post_lines)
    extra_after[-1] = ' '.join(continuation_last)

    records = []
    prev_balance = None

    for i, anchor in enumerate(anchors):
        description = f"{extra_before[i]} {anchor['mid_text']} {extra_after[i]}".strip()
        description = re.sub(r'\s+', ' ', description)

        if anchor['is_opening']:
            txn_type = 'opening_balance'
        elif prev_balance is None:
            txn_type = 'unknown'
        else:
            txn_type = 'credit' if anchor['balance'] > prev_balance else 'debit'

        records.append({
            'date': anchor['date'],
            'description': description,
            'amount': anchor['amount'],
            'balance': anchor['balance'],
            'txn_type': txn_type,
        })
        prev_balance = anchor['balance']

    return records


def parse_icici_savings_regular_statement(pdf_path: str) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())

    records = parse_icici_savings_regular_text(full_text)
    df = pd.DataFrame(records)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['source'] = 'icici_savings'
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python icici_savings_regular.py <path_to_statement.pdf>")
        sys.exit(1)

    with pdfplumber.open(sys.argv[1]) as pdf:
        num_pages = len(pdf.pages)

    df = parse_icici_savings_regular_statement(sys.argv[1])
    print(df.to_string())
    print(f"\nPages in PDF: {num_pages}")
    print(f"Total transactions parsed: {len(df)}")
    if 'txn_type' in df.columns and (df['txn_type'] == 'unknown').any():
        print(f"NOTE: {(df['txn_type']=='unknown').sum()} row(s) have unknown debit/credit type.")