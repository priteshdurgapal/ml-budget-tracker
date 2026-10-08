"""Debug helper: shows raw extracted text AND what got parsed, side by side,
so we can see exactly which rows are being dropped and why."""

import sys
import pdfplumber
from ic_parser import parse_icici_cc_page_text, ROW_START_PATTERN

if len(sys.argv) < 2:
    print("Usage: python debug_icici.py <path_to_statement.pdf>")
    sys.exit(1)

with pdfplumber.open(sys.argv[1]) as pdf:
    full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

print("=" * 80)
print("RAW EXTRACTED TEXT (all pages concatenated):")
print("=" * 80)
print(full_text)

print("\n" + "=" * 80)
print("ROW-START MATCHES FOUND (date + serial number pattern):")
print("=" * 80)
matches = list(ROW_START_PATTERN.finditer(full_text))
print(f"Found {len(matches)} row-start matches")
for m in matches:
    print(f"  date={m.group(1)}  serial={m.group(2)}")

print("\n" + "=" * 80)
print("PARSED TRANSACTIONS:")
print("=" * 80)
results = parse_icici_cc_page_text(full_text)
print(f"Parsed {len(results)} transactions")
for r in results:
    print(f"  {r['date']}  {r['serial_no']}  {r['description'][:40]:40}  {r['amount']}  {r['txn_type']}")