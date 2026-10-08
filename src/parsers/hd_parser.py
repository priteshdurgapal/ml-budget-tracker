"""
Parser for HDFC Bank Credit Card statements.
extract_tables() doesn't work reliably for this layout (no clean grid lines),
so we parse extract_text() directly using regex.
"""

import re
import pandas as pd
import pdfplumber

# Matches "12/05/2026| 00:00" style date-time markers
DATE_TIME_PATTERN = re.compile(r'(\d{2}/\d{2}/\d{4})\|\s*(\d{2}:\d{2})')

# Rewards column: "+ 10" style, directly followed by "C <amount>" - digits present
REWARDS_PATTERN = re.compile(r'\+\s*(\d+)\s*C\s*([\d,]+\.\d{2})')

# Credit indicator: bare "+" with NOTHING (no digits) between it and "C <amount>"
CREDIT_PATTERN = re.compile(r'\+\s*C\s*([\d,]+\.\d{2})')

# Plain debit, no plus sign at all before "C <amount>"
PLAIN_AMOUNT_PATTERN = re.compile(r'C\s*([\d,]+\.\d{2})')

# Cleanup: strip wrapped "(Ref# ...)" fragments that sometimes leak in from
# adjacent lines due to PDF text-extraction reordering around page columns
REF_CLEANUP_PATTERN = re.compile(r'\(Ref#[^)]*\)?')


def _extract_amount_and_type(block: str):
    """
    Returns (match_object, amount, rewards_points, txn_type) using priority:
    1. '+ <digits> C <amount>'  -> debit, with reward points
    2. '+ C <amount>' (bare +)  -> credit, 0 reward points
    3. 'C <amount>' (no +)      -> debit, 0 reward points
    """
    m = REWARDS_PATTERN.search(block)
    if m:
        amount = float(m.group(2).replace(',', ''))
        return m, amount, int(m.group(1)), 'debit'

    m = CREDIT_PATTERN.search(block)
    if m:
        amount = float(m.group(1).replace(',', ''))
        return m, amount, 0, 'credit'

    m = PLAIN_AMOUNT_PATTERN.search(block)
    if m:
        amount = float(m.group(1).replace(',', ''))
        return m, amount, 0, 'debit'

    return None, None, None, None


# Known tag prefixes HDFC prepends to the merchant description with no separator
# (add more here as you discover them in your actual statements)
KNOWN_TAGS = [
    #'EMI ADHOC',
    #'EMI PROCESSING FEE',
    'EMI'#,
    #'REVERSAL',
    #'REFUND',
]
TAG_PATTERN = re.compile(
    r'^(' + '|'.join(re.escape(t) for t in KNOWN_TAGS) + r')\s*',
    re.IGNORECASE
)


def _split_tag(description: str):
    """Split a known tag prefix off the front of the description, if present."""
    m = TAG_PATTERN.match(description)
    if m:
        tag = m.group(1).strip()
        merchant = description[m.end():].strip()
        return tag, merchant
    return None, description


def parse_hdfc_cc_page_text(text: str) -> list[dict]:
    """Parse one page's raw text into a list of transaction dicts."""

    # Transactions only start after this header line
    if "DATE & TIME" in text:
        text = text.split("DATE & TIME", 1)[1]
    else:
        return []  # this page has no transaction table (e.g. summary page)

    matches = list(DATE_TIME_PATTERN.finditer(text))
    transactions = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        amt_match, amount, rewards, txn_type = _extract_amount_and_type(block)
        if amt_match is None:
            continue  # couldn't find an amount in this block, skip

        # description = everything in the block before the matched amount pattern
        description = block[:amt_match.start()].strip()
        description = REF_CLEANUP_PATTERN.sub('', description)  # strip stray Ref# fragments
        description = re.sub(r'\s+', ' ', description).strip()

        tag, merchant = _split_tag(description)

        transactions.append({
            'date': m.group(1),
            'time': m.group(2),
            'tag': tag,
            'description': merchant,
            'amount': amount,
            'rewards_points': rewards,
            'txn_type': txn_type,
        })

    return transactions


def parse_hdfc_cc_statement(pdf_path: str) -> pd.DataFrame:
    """Parse a full HDFC CC statement PDF into a DataFrame."""
    all_transactions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_transactions.extend(parse_hdfc_cc_page_text(text))

    df = pd.DataFrame(all_transactions)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df['source'] = 'hdfc_cc'
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hdfc_cc.py <path_to_statement.pdf>")
        sys.exit(1)

    df = parse_hdfc_cc_statement(sys.argv[1])
    print(df.to_string())
    print(f"\nTotal transactions parsed: {len(df)}")