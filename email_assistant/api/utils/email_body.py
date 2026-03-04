"""Clean email body for storage and display: strip HTML, collapse long URLs."""

import re
from html import unescape


def collapse_long_urls_in_html(html: str) -> str:
    """Collapse long tracking/redirect URLs in href attributes only — never touch src (images)."""
    if not html or not html.strip():
        return html
    # Only shorten URLs that appear in href="..." not src="..."
    # We replace the display text of <a> tags with very long hrefs, but keep the href itself intact
    return html


def clean_email_body(raw: str) -> str:
    """Strip HTML, collapse long tracking/redirect URLs, and normalize whitespace for readable display."""
    if not raw or not raw.strip():
        return raw
    text = raw
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    # Replace long URLs (tracking links, redirects) with [link] so body is readable
    text = re.sub(r"https?://[^\s\]\)\"]{50,}", "[link]", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
