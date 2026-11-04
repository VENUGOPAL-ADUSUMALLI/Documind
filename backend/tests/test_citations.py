from app.citations import check_citations

SOURCES = [
    {"filename": "report.pdf", "page": 3},
    {"filename": "report.pdf", "page": 6},
    {"filename": "my notes.pdf", "page": 1},
]


def test_valid_citations_are_left_untouched():
    answer = "A is true [report.pdf, page 3]. B is true [my notes.pdf, page 1]."
    cleaned, invalid = check_citations(answer, SOURCES)
    assert cleaned == answer
    assert invalid == []


def test_page_that_was_never_retrieved_is_removed():
    cleaned, invalid = check_citations("Claim [report.pdf, page 9].", SOURCES)
    assert cleaned == "Claim."
    assert invalid == ["report.pdf, page 9"]


def test_right_page_but_wrong_file_is_invalid():
    cleaned, invalid = check_citations("Claim [other.pdf, page 3].", SOURCES)
    assert "other.pdf" not in cleaned
    assert invalid == ["other.pdf, page 3"]


def test_mixed_valid_and_invalid_and_duplicates():
    answer = "One [report.pdf, page 6]. Two [report.pdf, page 10]. Three [report.pdf, page 10]."
    cleaned, invalid = check_citations(answer, SOURCES)
    assert cleaned == "One [report.pdf, page 6]. Two. Three."
    assert invalid == ["report.pdf, page 10"]  # reported once


def test_answer_without_citations_is_unchanged():
    cleaned, invalid = check_citations("No citations here, just [a note] in brackets.", SOURCES)
    assert cleaned == "No citations here, just [a note] in brackets."
    assert invalid == []
