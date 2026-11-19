

import re

CITATION = re.compile(r"\s?\[([^\[\]]+?),\s*page\s+(\d+)\]")


def check_citations(answer, sources):
    allowed = {(source["filename"], source["page"]) for source in sources}
    invalid = []

    def keep_or_remove(match):
        filename = match.group(1).strip()
        page = int(match.group(2))
        if (filename, page) in allowed:
            return match.group(0)

        label = f"{filename}, page {page}"
        if label not in invalid:
            invalid.append(label)
        return ""  # invalid: remove it from the answer

    cleaned = CITATION.sub(keep_or_remove, answer)
    return cleaned, invalid
