from sec_filing_analysis.data.preprocessor import DocumentChunker, SecFilingParser

FILLER = "Net sales grew during the period. " * 40

FORM_10K = (
    # table of contents
    "Item 1. Business 1 Item 1A. Risk Factors 5 Item 1B. Unresolved Staff Comments 17 "
    "Item 7. Management's Discussion and Analysis of Financial Condition 20 "
    "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 27 "
    "Item 8. Financial Statements and Supplementary Data 28 "
    "Item 9. Changes in and Disagreements With Accountants 50 "
    # body
    "Item 1. Business The Company designs products. " + FILLER +
    "Item 1A. Risk Factors This should be read together with Part II, Item 7, "
    "“Management’s Discussion and Analysis”. Supply risk. " + FILLER +
    "Item 1B. Unresolved Staff Comments None. "
    "Item 7. Management’s Discussion and Analysis of Financial Condition MDA_BODY " + FILLER +
    "Item 7A. Quantitative and Qualitative Disclosures About Market Risk Rates. "
    "Item 8. Financial Statements and Supplementary Data STATEMENTS_BODY " + FILLER +
    "Item 9. Changes in and Disagreements With Accountants None."
)


class WhitespaceTokenizer:
    """Stand-in for the embedding tokenizer: one token per whitespace-separated word."""

    class Encoding(dict):
        def word_ids(self):
            return list(range(len(self["offset_mapping"])))

    def __call__(self, text, **kwargs):
        offsets, position = [], 0
        for word in text.split():
            start = text.index(word, position)
            position = start + len(word)
            offsets.append((start, position))
        return self.Encoding(offset_mapping=offsets)


def test_sections_come_from_the_body_not_the_table_of_contents():
    sections = SecFilingParser().extract_sections(FORM_10K, "10-K")

    assert set(sections) == {"BUSINESS", "RISK_FACTORS", "MD&A", "FINANCIAL_STATEMENTS"}
    assert "MDA_BODY" in sections["MD&A"]
    assert "STATEMENTS_BODY" in sections["FINANCIAL_STATEMENTS"]


def test_section_ends_at_the_next_item_heading():
    sections = SecFilingParser().extract_sections(FORM_10K, "10-K")

    assert "Item 7A" not in sections["MD&A"]
    assert "STATEMENTS_BODY" not in sections["MD&A"]


def test_cross_reference_does_not_start_a_section():
    sections = SecFilingParser().extract_sections(FORM_10K, "10-K")

    # Risk Factors mentions 'Item 7, "Management's Discussion..."' in passing
    assert sections["MD&A"].startswith("Item 7. Management")
    assert "Supply risk" in sections["RISK_FACTORS"]


def test_10q_md_and_a_is_part_one_item_two():
    text = (
        "Item 1. Financial Statements Balance sheet. " + FILLER +
        "Item 2. Management’s Discussion and Analysis of Financial Condition MDA_BODY " + FILLER +
        "Item 3. Quantitative and Qualitative Disclosures About Market Risk None. "
        "Item 1A. Risk Factors Risks. " + FILLER +
        "Item 2. Unregistered Sales of Equity Securities BUYBACK_TABLE " + FILLER +
        "Item 6. Exhibits"
    )

    sections = SecFilingParser().extract_sections(text, "10-Q")

    assert "MDA_BODY" in sections["MD&A"]
    assert "BUYBACK_TABLE" not in sections["MD&A"]


def test_text_extraction_uses_main_document_and_keeps_split_words(tmp_path):
    (tmp_path / "full-submission.txt").write_text("<DOCUMENT><TEXT>EXHIBIT ONLY</TEXT></DOCUMENT>")
    (tmp_path / "primary-document.html").write_text(
        "<html><body><ix:header>us-gaap:HiddenMember</ix:header>"
        "<div><span>ITEM 1A. RIS</span><span>K FACTORS</span></div><div>Next paragraph</div>"
        "<table><tr><td>Revenue</td><td>100</td></tr></table></body></html>"
    )

    text = SecFilingParser().extract_text_from_filing(tmp_path / "full-submission.txt")

    assert text == "ITEM 1A. RISK FACTORS Next paragraph Revenue | 100"


def test_chunks_respect_token_limit_and_overlap():
    words = [f"w{i}" for i in range(250)]
    chunker = DocumentChunker(chunk_size=102, chunk_overlap=20, tokenizer=WhitespaceTokenizer())

    chunks = [chunk.split() for chunk in chunker.chunk_text(" ".join(words))]

    assert all(len(chunk) <= 100 for chunk in chunks)  # 102 minus [CLS] and [SEP]
    assert chunks[0] == words[:100]
    assert chunks[1][:20] == chunks[0][-20:]
    assert chunks[-1][-1] == words[-1]
    assert [w for chunk in chunks for w in chunk[20:]] == words[20:]
