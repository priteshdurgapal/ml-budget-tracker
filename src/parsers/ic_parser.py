"""
Parser for ICICI Bank Credit Card statements.
Layout: Date | SerNo. | Transaction Details | Reward Points | Intl.# amount | Amount (in `)
Credits are marked with a trailing "CR" after the amount (debits have no suffix).
"""

import re
import pandas as pd
import pdfplumber

# Each transaction row starts with "DD/MM/YYYY <serial_number>"
ROW_START_PATTERN = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(\d{6,})\s+')

# Masked card number header, e.g. "4035XXXXXXXX6001" - appears before each card's section
CARD_HEADER_PATTERN = re.compile(r'^(\d{4}X+\d{4})$', re.MULTILINE)

# The amount is the LAST decimal-formatted number (X,XXX.XX) in a block.
# We don't anchor this to the end of the block, because wrapped descriptions
# or trailing footnote text can appear AFTER the true amount in the raw
# extracted text (a pdfplumber column-ordering quirk).
AMOUNT_PATTERN = re.compile(r'([\d,]+\.\d{2})\s*(CR)?')


# Markers that signal the END of the transaction table and start of
# boilerplate (T&Cs, MITC illustrations, reward program text, etc).
# We truncate the text here BEFORE parsing so multi-page statements with
# lots of trailing legal/illustration text don't get swallowed into the
# last transaction's description or produce spurious fake matches.
TRANSACTION_SECTION_END_MARKERS = [
    r'#\s*International Spends',
    r'IMPORTANT MESSAGES',
    r'GREAT OFFERS ON YOUR CARD',
    r'ICICI\S*\s*Bank Rewards',
]


def _truncate_to_transaction_section(text: str) -> str:
    """Cut off text at the earliest known end-of-table marker, if found."""
    earliest = None
    for pattern in TRANSACTION_SECTION_END_MARKERS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()
    return text[:earliest] if earliest is not None else text


def parse_icici_cc_page_text(text: str) -> list[dict]:
    """Parse one page's raw text into a list of transaction dicts."""

    text = _truncate_to_transaction_section(text)

    transactions = []

    current_card = None
    card_positions = [(m.start(), m.group(1)) for m in CARD_HEADER_PATTERN.finditer(text)]

    matches = list(ROW_START_PATTERN.finditer(text))

    for i, m in enumerate(matches):
        for pos, card_num in card_positions:
            if pos <= m.start():
                current_card = card_num

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        amount_matches = list(AMOUNT_PATTERN.finditer(block))
        if not amount_matches:
            continue  # genuinely no amount in this block - skip (e.g. stray text)

        # take the LAST decimal-amount match in the block as the true amount
        amt_match = amount_matches[-1]
        amount = float(amt_match.group(1).replace(',', ''))
        txn_type = 'credit' if amt_match.group(2) == 'CR' else 'debit'

        pre_text = block[:amt_match.start()]

        # try to pull a trailing standalone digit as reward points (optional -
        # not critical for expense analysis, so we don't fail if it's not there)
        reward_match = re.search(r'(?<!\S)(\d+)\s*$', pre_text)
        if reward_match:
            reward_points = int(reward_match.group(1))
            description = pre_text[:reward_match.start()]
        else:
            reward_points = 0
            description = pre_text

        description = re.sub(r'\s+', ' ', description).strip()

        transactions.append({
            'date': m.group(1),
            'serial_no': m.group(2),
            'card_last4': current_card,
            'description': description,
            'reward_points': reward_points,
            'amount': amount,
            'txn_type': txn_type,
        })

    return transactions


def parse_icici_cc_statement(pdf_path: str) -> pd.DataFrame:
    """Parse a full ICICI CC statement PDF into a DataFrame."""

    # Concatenate all pages into ONE text block before parsing.
    # This matters because a transaction row can be split across a page
    # boundary (date+serial on page N, amount on page N+1) - parsing
    # page-by-page would silently drop such rows.
    full_text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text_parts.append(text)

    full_text = "\n".join(full_text_parts)
    all_transactions = parse_icici_cc_page_text(full_text)

    df = pd.DataFrame(all_transactions)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df['source'] = 'icici_cc'
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python icici_cc.py <path_to_statement.pdf>")
        sys.exit(1)

    with pdfplumber.open(sys.argv[1]) as pdf:
        num_pages = len(pdf.pages)

    df = parse_icici_cc_statement(sys.argv[1])
    print(df.to_string())
    print(f"\nPages in PDF: {num_pages}")
    print(f"Total transactions parsed: {len(df)}")
    print("^ Manually cross-check this count against the PDF before trusting the output.")