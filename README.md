# Cutting-Edge AI PDF Summarizer

A Streamlit application that turns complex PDF papers into concise, actionable insights using Groq and `openai/gpt-oss-20b`.

## Project Design

1. **Interface:** Image-led hero banner, right-aligned PDF uploader, collapsible settings, format selection, and document metrics.
2. **PDF processing:** `pypdf` extracts embedded text. Scanned pages are rendered with `pypdfium2` and read with Tesseract OCR through `pytesseract`.
3. **AI summarization:** Extracted text and optional instructions are sent to Groq through LangChain. Output supports concise, detailed, bullet-point, bar-chart, and pie-chart formats.
4. **Statistics:** Labeled numeric values such as `Revenue: 1200` are detected, shown in a table, and visualized with Plotly.
5. **Executive export:** Summaries, metrics, and statistics can be downloaded as text or as a print-ready PDF generated with ReportLab.
6. **Configuration:** `.streamlit/config.toml` allows uploads up to 200 MB. API credentials are loaded from `.env` using `GROQ_API_KEY`.

## Run Locally

```bash
git clone https://github.com/KamRachakonda/Cutting-Edge-AI-PDF-Summarizer.git
cd Cutting-Edge-AI-PDF-Summarizer
pip install -r requirements.txt
streamlit run app.py
```

For scanned PDFs on macOS, install Tesseract if needed:

```bash
brew install tesseract
```

Create `.env` with:

```env
GROQ_API_KEY=your_api_key_here
```

## Main Files

- `app.py` - Interface, OCR workflow, charts, and exports.
- `pdf_processor.py` - PDF summarization service.
- `summerizer.py` - Alternate summarization implementation.
- `utility.py` - Configuration, validation, logging, caching, and errors.
- `pdf-summarizer-banner.png` - Hero image.
- `requirements.txt` - Python dependencies.