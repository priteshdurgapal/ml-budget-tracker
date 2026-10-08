"""
Parser for YES Bank Credit Card statements.
This PDF has an actual bordered/gridded table, so we use pdfplumber's
extract_tables() directly instead of regex on raw text - far more reliable
since it respects the PDF's real column structure regardless of text
reading-order quirks.
"""

import re
import pandas as pd
import pdfplumber

AMOUNT_PATTERN = re.compile(r'([\d,]+\.\d{2})\s*(Dr|Cr)', re.IGNORECASE)
DATE_PATTERN = re.compile(r'^\d{2}/\d{2}/\d{4}$')


def _clean_cell(value):
    """Collapse newlines/extra whitespace within a table cell."""
    if not value:
        return None
    cleaned = re.sub(r'\s+', ' ', value).strip()
    return cleaned if cleaned else None


def parse_yesbank_cc_statement(pdf_path: str) -> pd.DataFrame:
    """Parse a full YES Bank CC statement PDF into a DataFrame."""
    records = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue

                header = table[0]
                if not header or len(header) < 4:
                    continue
                if not (header[0] and 'Date' in header[0]):
                    continue
                if not (header[1] and 'Transaction' in header[1]):
                    continue

                for row in table[1:]:
                    if not row or len(row) < 4:
                        continue
                    date_str, desc_raw, category_raw, amount_raw = row[:4]
                    if not date_str or not DATE_PATTERN.match(date_str.strip()):
                        continue

                    amt_match = AMOUNT_PATTERN.search(amount_raw or '')
                    if not amt_match:
                        continue

                    amount = float(amt_match.group(1).replace(',', ''))
                    txn_type = 'debit' if amt_match.group(2).lower() == 'dr' else 'credit'

                    records.append({
                        'date': date_str.strip(),
                        'description': _clean_cell(desc_raw),
                        'merchant_category': _clean_cell(category_raw),
                        'amount': amount,
                        'txn_type': txn_type,
                    })

    df = pd.DataFrame(records)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df['source'] = 'yesbank_cc'
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python yesbank_cc.py <path_to_statement.pdf>")
        sys.exit(1)

    with pdfplumber.open(sys.argv[1]) as pdf:
        num_pages = len(pdf.pages)

    df = parse_yesbank_cc_statement(sys.argv[1])
    print(df.to_string())
    print(f"\nPages in PDF: {num_pages}")
    print(f"Total transactions parsed: {len(df)}")
    print("^ Manually cross-check this count against the PDF before trusting the output.")