"""Check that every citation in an answer points to a page we actually gave the model.

The prompt asks the model to cite like  [report.pdf, page 3]  but a prompt is only
a request: the model can still write a page that does not exist. So the code checks.

What this catches:   a (file, page) that was NOT among the chunks sent to the model.
What it can't catch: a real page attached to a claim that page does not support.
"""

import re

# Matches  [some file.pdf, page 12]  and also the single space before it.
CITATION = re.compile(r"\s?\[([^\[\]]+?),\s*page\s+(\d+)\]")


def check_citations(answer, sources):
    """Returns (cleaned_answer, invalid_citations).

    Valid citations stay in the text. Invalid ones are removed from the text and
    listed (as 'file.pdf, page 9') so the caller can warn the user.
    """
    allowed = {(source["filename"], source["page"]) for source in sources}
    invalid = []

    def keep_or_remove(match):
        filename = match.group(1).strip()
        page = int(match.group(2))
        if (filename, page) in allowed:
            return match.group(0)  # valid: leave the text unchanged

        label = f"{filename}, page {page}"
        if label not in invalid:
            invalid.append(label)
        return ""  # invalid: remove it from the answer

    cleaned = CITATION.sub(keep_or_remove, answer)
    return cleaned, invalid
