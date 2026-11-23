"""Text and tone of earnings press releases.

The press release is what the market reads when results are announced. It is
attached to the 8-K as exhibit 99.1.
"""

import re

import pandas as pd

from ..data.preprocessor import DocumentChunker, SecFilingParser
from ..utils.logger import log
from . import features
from .events import sec_get

SUBMISSION_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{accession}.txt"
_EXHIBIT = re.compile(rb"<DOCUMENT>\s*<TYPE>(EX-99[^\s<]*).*?<TEXT>(.*?)</TEXT>", re.DOTALL)

RELEASE_SOURCE = "a company's quarterly earnings press release"

# Press releases are short; some consist of a few paragraphs and tables.
MIN_RELEASE_SENTENCES = 10
TONE_COLUMNS = ["pr_kw_tone", "pr_finbert_tone", "pr_llm_tone"]


def extract_press_release(submission: bytes) -> str:
    """Text of exhibit 99.1 of an 8-K submission ("" if there is none).

    Falls back to the first other exhibit 99 if no 99.1 exists.
    """
    exhibits = _EXHIBIT.findall(submission)
    if not exhibits:
        return ""

    preferred = [html for kind, html in exhibits if kind in (b"EX-99.1", b"EX-99.01")]
    return SecFilingParser().html_to_text(preferred[0] if preferred else exhibits[0][1])


def fetch_press_releases(releases: pd.DataFrame) -> pd.DataFrame:
    """Download the press release of each earnings 8-K (rows: release_accession, text)."""
    releases = releases.dropna(subset=["release_accession"]).drop_duplicates("release_accession")
    rows = []

    for count, release in enumerate(releases.itertuples(), start=1):
        url = SUBMISSION_URL.format(
            cik=int(release.cik),
            folder=release.release_accession.replace("-", ""),
            accession=release.release_accession,
        )
        try:
            text = extract_press_release(sec_get(url).content)
        except Exception as e:
            log.warning(f"Could not fetch {release.release_accession}: {e}")
            text = ""
        rows.append({"release_accession": release.release_accession, "text": text})
        if count % 50 == 0:
            log.info(f"Press releases: {count}/{len(releases)}")

    return pd.DataFrame(rows)


def build_release_features(press_releases: pd.DataFrame, releases: pd.DataFrame, use_llm: bool = True) -> pd.DataFrame:
    """Tone of each press release and its change against the company's previous one.

    `releases` needs ticker, release_accession and release_accepted.
    """
    text_by_release = press_releases.set_index("release_accession")["text"].to_dict()
    sentences = {accession: features.split_sentences([text]) for accession, text in text_by_release.items()}
    usable = {accession for accession, found in sentences.items() if len(found) >= MIN_RELEASE_SENTENCES}
    log.info(f"{len(usable)} of {len(text_by_release)} press releases have enough prose to score")
    sentences = {accession: sentences[accession] for accession in usable}

    result = releases[["ticker", "release_accession", "release_accepted"]].drop_duplicates("release_accession").copy()
    result["pr_sentences"] = result["release_accession"].map({k: len(v) for k, v in sentences.items()}).fillna(0).astype(int)
    result["pr_kw_tone"] = result["release_accession"].map({k: features.keyword_tone(v) for k, v in sentences.items()})
    result["pr_finbert_tone"] = result["release_accession"].map(features.finbert_tone(sentences))

    if use_llm:
        chunker = DocumentChunker()
        chunks = pd.DataFrame([
            {"accession": accession, "source": RELEASE_SOURCE, "text": chunk}
            for accession in sorted(usable)
            for chunk in chunker.chunk_text(text_by_release[accession])
        ])
        result["pr_llm_tone"] = result["release_accession"].map(features.llm_tone(chunks))

    result = result.sort_values(["ticker", "release_accepted"]).reset_index(drop=True)
    for column in TONE_COLUMNS:
        if column in result:
            result[f"d_{column}"] = result[column] - result.groupby("ticker")[column].shift(1)

    return result.drop(columns=["ticker", "release_accepted"])
