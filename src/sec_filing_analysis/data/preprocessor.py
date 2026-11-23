from pathlib import Path
from dataclasses import dataclass
import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import pandas as pd
from ..config import settings
from ..utils.logger import log


@dataclass
class Document:
    ticker: str
    filing_type: str
    filing_date: str
    section: str
    text: str
    metadata: dict


# Separators that may sit between an item number and its title once the HTML
# has been flattened to text ("Item 7. | Management's ...").
_SEP = r"[\s|.:\-–—]*"


def _heading(item: str, title: str) -> re.Pattern:
    """Pattern for a section heading: the item number followed by its title.

    Requiring the title keeps cross-references such as "see Part II, Item 7"
    from being mistaken for the start of a section.
    """
    return re.compile(rf"Item\s*{item}{_SEP}{title}", re.IGNORECASE)


class SecFilingParser:
    # section name -> (start heading, end heading). The end heading is the
    # item that follows the section in the form's fixed order.
    SECTION_HEADINGS = {
        "10-K": {
            "BUSINESS": (
                _heading("1", r"Business"),
                _heading("1A", r"Risk\s+Factors"),
            ),
            "RISK_FACTORS": (
                _heading("1A", r"Risk\s+Factors"),
                _heading("(?:1B|2)", r"(?:Unresolved\s+Staff|Properties)"),
            ),
            "MD&A": (
                _heading("7", r"Management\W{0,3}s\s+Discussion"),
                _heading("(?:7A|8)", r"(?:Quantitative\s+and\s+Qualitative|Financial\s+Statements)"),
            ),
            "FINANCIAL_STATEMENTS": (
                _heading("8", r"Financial\s+Statements"),
                _heading("9", r"Changes\s+in\s+and\s+Disagreements"),
            ),
        },
        "10-Q": {
            "FINANCIAL_STATEMENTS": (
                _heading("1", r"(?:Condensed\s+)?(?:Consolidated\s+)?Financial\s+Statements"),
                _heading("2", r"Management\W{0,3}s\s+Discussion"),
            ),
            "MD&A": (
                _heading("2", r"Management\W{0,3}s\s+Discussion"),
                _heading("(?:3|4)", r"(?:Quantitative\s+and\s+Qualitative|Controls\s+and\s+Procedures)"),
            ),
            "RISK_FACTORS": (
                _heading("1A", r"Risk\s+Factors"),
                _heading("(?:2|6)", r"(?:Unregistered\s+Sales|Exhibits)"),
            ),
        },
    }

    # Table-of-contents entries match the heading patterns too, but the span
    # up to the next entry is only a line long.
    MIN_SECTION_CHARS = 500

    BLOCK_TAGS = ['p', 'div', 'li', 'tr', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']

    def _read_main_document(self, file_path: Path) -> bytes:
        """Return the HTML of the 10-K/10-Q itself, without exhibits.

        full-submission.txt bundles every exhibit and XBRL file of the filing.
        sec-edgar-downloader stores the main document next to it; if it is
        missing, fall back to the first <TEXT> block of the submission.
        """
        primary = file_path.with_name("primary-document.html")
        if primary.exists():
            return primary.read_bytes()

        raw = file_path.read_bytes()
        match = re.search(rb"<TEXT>(.*?)</TEXT>", raw, re.DOTALL)
        return match.group(1) if match else raw

    def extract_text_from_filing(self, file_path: Path) -> str:
        try:
            return self.html_to_text(self._read_main_document(file_path))
        except Exception as e:
            log.error(f"Failed to extract text from {file_path}: {e}")
            return ""

    def html_to_text(self, html: bytes) -> str:
        """Flatten an EDGAR HTML document to one line of text; table cells are joined with ' | '."""
        with warnings.catch_warnings():
            # inline-XBRL filings declare themselves as XML but are HTML
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html, 'lxml')

        # ix:header holds hidden inline-XBRL context definitions, not prose
        for tag in soup(['script', 'style', 'ix:header']):
            tag.decompose()

        # Put whitespace after block elements only. Filings often split a
        # single word across inline <span>/<font> tags ("RIS" + "K FACTORS"),
        # so joining every text node with a space would break words apart.
        for tag in soup.find_all(self.BLOCK_TAGS):
            tag.append(' ')
        for br in soup.find_all('br'):
            br.replace_with(' ')

        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [' '.join(td.get_text().split()) for td in tr.find_all(['td', 'th'])]
                cells = [cell for cell in cells if cell]
                if cells:
                    rows.append(' | '.join(cells))
            table.replace_with(soup.new_string(' ' + ' '.join(rows) + ' '))

        return re.sub(r'\s+', ' ', soup.get_text()).strip()

    def extract_sections(self, text: str, filing_type: str) -> dict[str, str]:
        sections = {}

        for section_name, (start_re, end_re) in self.SECTION_HEADINGS.get(filing_type, {}).items():
            ends = [m.start() for m in end_re.finditer(text)]
            span = None

            # The same heading appears in the table of contents and in the
            # body. Keep the last occurrence that is followed by real content.
            for start in (m.start() for m in start_re.finditer(text)):
                end = next((e for e in ends if e > start), None)
                if end is not None and end - start >= self.MIN_SECTION_CHARS:
                    span = (start, end)

            if span:
                sections[section_name] = text[span[0]:span[1]].strip()

        return sections

    def parse_filing(self, file_path: Path, ticker: str, filing_type: str, filing_date: str) -> list[Document]:
        text = self.extract_text_from_filing(file_path)
        if not text:
            return []

        sections = self.extract_sections(text, filing_type)
        if not sections:
            log.warning(f"No sections found in {file_path}")

        return [
            Document(
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                section=section_name,
                text=section_text,
                metadata={
                    "file_path": str(file_path),
                    "text_length": len(section_text),
                },
            )
            for section_name, section_text in sections.items()
        ]


class DocumentChunker:
    """Splits text into chunks of at most `chunk_size` embedding-model tokens.

    Sizes are measured with the embedding model's own tokenizer so that no
    chunk exceeds the model's input limit; anything beyond that limit would be
    cut off silently when the chunk is embedded.
    """

    SPECIAL_TOKENS = 2  # [CLS] and [SEP] are added at embedding time

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, tokenizer=None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        return self._tokenizer

    def chunk_text(self, text: str) -> list[str]:
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoding["offset_mapping"]
        word_ids = encoding.word_ids()
        n_tokens = len(offsets)

        def word_start(index: int, lower_bound: int) -> int:
            """Move a cut point back so it does not fall inside a word."""
            cut = index
            while cut > lower_bound and word_ids[cut] == word_ids[cut - 1]:
                cut -= 1
            return cut if cut > lower_bound else index

        window = self.chunk_size - self.SPECIAL_TOKENS
        chunks = []
        start = 0

        while start < n_tokens:
            end = min(start + window, n_tokens)
            if end < n_tokens:
                end = word_start(end, start)

            chunk = text[offsets[start][0]:offsets[end - 1][1]].strip()
            if chunk:
                chunks.append(chunk)

            if end >= n_tokens:
                break
            start = word_start(max(end - self.chunk_overlap, start + 1), start)

        return chunks

    def chunk_document(self, doc: Document) -> list[dict]:
        chunks = self.chunk_text(doc.text)
        chunked_docs = []

        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                "ticker": doc.ticker,
                "filing_type": doc.filing_type,
                "filing_date": doc.filing_date,
                "section": doc.section,
                "chunk_id": idx,
                "text": chunk,
                "metadata": {
                    **doc.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            })

        return chunked_docs


def resolve_filing_path(stored_path: str) -> Path:
    """Locate a filing listed in the filing index.

    The index stores paths relative to the raw data directory. Indexes written
    by earlier versions hold absolute paths, which break as soon as the project
    folder is moved, so those are re-rooted at the current raw data directory.
    """
    path = Path(stored_path)
    if not path.is_absolute():
        return settings.raw_data_dir / path
    if not path.exists() and "sec-edgar-filings" in path.parts:
        return settings.raw_data_dir.joinpath(*path.parts[path.parts.index("sec-edgar-filings"):])
    return path


class DocumentProcessor:
    def __init__(self):
        self.parser = SecFilingParser()
        self.chunker = DocumentChunker()

    def process_filing(self, file_path: Path, ticker: str, filing_type: str, filing_date: str) -> list[dict]:
        documents = self.parser.parse_filing(file_path, ticker, filing_type, filing_date)

        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def process_batch(self, filing_index: pd.DataFrame) -> pd.DataFrame:
        all_chunks = []

        for _, row in filing_index.iterrows():
            try:
                file_path = resolve_filing_path(row["file_path"])
                chunks = self.process_filing(
                    file_path,
                    row["ticker"],
                    row["filing_type"],
                    str(row["filing_date"])
                )
                all_chunks.extend(chunks)
                log.info(f"Processed {row['ticker']} {row['filing_type']} {row['filing_date']}: {len(chunks)} chunks")

            except Exception as e:
                log.error(f"Failed to process {row['ticker']} {row['filing_type']}: {e}")

        df = pd.DataFrame(all_chunks)

        output_path = settings.processed_data_dir / "chunked_documents.parquet"
        df.to_parquet(output_path, index=False)

        log.info(f"Processed {len(df)} chunks from {len(filing_index)} filings")
        return df
