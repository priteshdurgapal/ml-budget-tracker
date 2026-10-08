# src/explore_pdf.py
import pdfplumber
import sys

def inspect(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        page = pdf.pages[0]
        print("--- RAW TEXT ---")
        print(page.extract_text())
        print("\n--- TABLES ---")
        for i, table in enumerate(page.extract_tables()):
            print(f"\nTable {i}:")
            for row in table[:5]:
                print(row)

if __name__ == "__main__":
    inspect(sys.argv[1])