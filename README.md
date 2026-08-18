# EMB-CAR Employee ID Card Generator

Automated batch generator for DENR-EMB Cordillera Administrative Region employee ID cards. Pulls employee data from a Google Form–linked spreadsheet, composites it onto a print-ready template, and outputs CR80-sized cards for the HiTi CS-2 dye-sublimation card printer.

## Features

- Batch-generates front and back card designs from spreadsheet rows
- Overlays employee photos with automatic crop/resize to fit the photo box
- Generates a unique Code128 barcode and QR code per employee
- Outputs at exact CR80 print dimensions (2.125 × 3.375 in) at 300 DPI
- Exports individual PNGs per employee, or a single print-ready multi-page PDF

## Tech Stack

- **Python 3.11**
- **Pillow** — image compositing
- **pandas** / **openpyxl** — spreadsheet reading
- **qrcode** — QR code generation
- **python-barcode** — Code128 barcode generation
- **img2pdf** — batching cards into a print-ready PDF
- **Jupyter Notebook** — generation workflow

## Prerequisites

- Python 3.10+ installed and added to PATH
- HiTi CS-2 driver installed, with card size set to CR80 in printer properties
- A Google Sheet (synced from the employee Google Form) exported as `.xlsx`

## Setup

```bash
# clone or download the project, then from the project folder:
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter notebook
```

## Project Structure

```
id-card-generator/
├── card_generator.ipynb     # main generation notebook
├── requirements.txt
├── .gitignore
├── README.md
├── templates/
│   ├── front_template.png   # 638x1013px, 300 DPI, CR80
│   └── back_template.png
├── fonts/
│   └── *.ttf                # bundled font(s) used on the card
├── data/                    # NOT committed -- see Data Privacy below
│   └── responses.xlsx
├── photos/                  # NOT committed -- employee ID photos
└── output/                  # NOT committed -- generated cards
    └── cards/
```

## Usage

1. Export the Google Sheet as `responses.xlsx` into `data/`.
2. Place employee photos in `photos/` (or configure the notebook to pull them from Google Drive links in the sheet).
3. Open `card_generator.ipynb` and run all cells.
4. Generated cards are written to `output/cards/` as individual PNGs, and optionally combined into a single print-ready PDF.

## Printing

- Printer: HiTi CS-2
- Card size (driver setting): CR80 / 2.125 × 3.375 in
- Print at **100% / Actual Size** — do not use "Fit to page," as this will distort the card dimensions
- Print one test card and measure it against a physical CR80 blank before running a full batch

## Data Privacy

This project handles personally identifiable information (full names, addresses, GSIS/TIN numbers, blood type, photos). The `data/`, `photos/`, and `output/` folders are excluded via `.gitignore` and must never be committed to version control. Keep this repository **private** if hosted on GitHub or any remote.

## Notes

Internal tool for DENR-EMB CAR, Baguio City. Not intended for external distribution.