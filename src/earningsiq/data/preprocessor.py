from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import re
from bs4 import BeautifulSoup
import pandas as pd
import polars as pl
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


class SecFilingParser:
    SECTION_PATTERNS = {
        "10-Q": {
            "MD&A": r"(?:ITEM\s*[2]\.|Management.*Discussion.*Analysis)",
            "RISK_FACTORS": r"(?:ITEM\s*[1]A\.|Risk\s*Factors)",
            "FINANCIAL_STATEMENTS": r"(?:ITEM\s*[1]\.|Financial\s*Statements)",
        },
        "10-K": {
            "MD&A": r"(?:ITEM\s*[7]\.|Management.*Discussion.*Analysis)",
            "RISK_FACTORS": r"(?:ITEM\s*[1]A\.|Risk\s*Factors)",
            "BUSINESS": r"(?:ITEM\s*[1]\.|Description.*Business)",
            "FINANCIAL_STATEMENTS": r"(?:ITEM\s*[8]\.|Financial\s*Statements)",
        }
    }

    def __init__(self):
        pass

    def extract_text_from_filing(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'lxml')
            for tag in soup(['script', 'style']):
                tag.decompose()

            table_texts = []
            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(' | '.join(cells))
                if rows:
                    table_texts.append('\n'.join(rows))
                table.replace_with(soup.new_string('\n' + '\n'.join(rows) + '\n' if rows else ''))

            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', '\n', text)

            return text

        except Exception as e:
            log.error(f"Failed to extract text from {file_path}: {e}")
            return ""

    def extract_sections(self, text: str, filing_type: str) -> dict[str, str]:
        sections = {}
        patterns = self.SECTION_PATTERNS.get(filing_type, {})

        toc_skip_threshold = 10000

        for section_name, pattern in patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))

            if matches:
                best_match = None
                best_content_length = 0

                for match in matches:
                    start_pos = match.start()

                    if start_pos < toc_skip_threshold and len(matches) > 1:
                        continue

                    next_item_pattern = r"ITEM\s*\d+[A-Z]?\."
                    search_start = start_pos + 100
                    next_matches = list(re.finditer(next_item_pattern, text[search_start:], re.IGNORECASE))

                    if next_matches:
                        end_pos = search_start + next_matches[0].start()
                    else:
                        end_pos = min(start_pos + 100000, len(text))

                    content_length = end_pos - start_pos

                    if content_length > best_content_length:
                        best_content_length = content_length
                        best_match = (start_pos, end_pos)

                if best_match is None and matches:
                    start_pos = matches[0].start()
                    next_item_pattern = r"ITEM\s*\d+[A-Z]?\."
                    search_start = start_pos + 100
                    next_matches = list(re.finditer(next_item_pattern, text[search_start:], re.IGNORECASE))
                    if next_matches:
                        end_pos = search_start + next_matches[0].start()
                    else:
                        end_pos = min(start_pos + 100000, len(text))
                    best_match = (start_pos, end_pos)

                if best_match:
                    start_pos, end_pos = best_match
                    section_text = text[start_pos:end_pos]
                    if len(section_text) > 500:
                        sections[section_name] = section_text[:50000]

        return sections

    def parse_filing(self, file_path: Path, ticker: str, filing_type: str, filing_date: str) -> list[Document]:
        text = self.extract_text_from_filing(file_path)
        if not text:
            return []

        sections = self.extract_sections(text, filing_type)
        documents = []

        for section_name, section_text in sections.items():
            if len(section_text.strip()) > 100:
                doc = Document(
                    ticker=ticker,
                    filing_type=filing_type,
                    filing_date=filing_date,
                    section=section_name,
                    text=section_text,
                    metadata={
                        "file_path": str(file_path),
                        "text_length": len(section_text),
                    }
                )
                documents.append(doc)

        return documents


class DocumentChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)

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
                file_path = Path(row["file_path"])
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

    def process_batch_parallel(self, filing_index: pd.DataFrame, n_partitions: int = 4) -> pd.DataFrame:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar

        ddf = dd.from_pandas(filing_index, npartitions=n_partitions)

        def process_row(row):
            try:
                file_path = Path(row["file_path"])
                chunks = self.process_filing(
                    file_path,
                    row["ticker"],
                    row["filing_type"],
                    str(row["filing_date"])
                )
                return pd.DataFrame(chunks)
            except Exception as e:
                log.error(f"Error processing {row['ticker']}: {e}")
                return pd.DataFrame()

        with ProgressBar():
            results = ddf.apply(process_row, axis=1, meta=pd.DataFrame()).compute()

        df = pd.concat([r for r in results if not r.empty], ignore_index=True)

        output_path = settings.processed_data_dir / "chunked_documents.parquet"
        df.to_parquet(output_path, index=False)

        log.info(f"Processed {len(df)} chunks from {len(filing_index)} filings in parallel")
        return df
