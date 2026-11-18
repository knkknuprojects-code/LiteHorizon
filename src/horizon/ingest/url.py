"""Plain-URL source — fetch web pages and extract their main text.

The highest-noise, highest-burden adapter (C1 ranked it last): arbitrary web pages
carry nav/ads/boilerplate and come with robots.txt / ToS / rate-limit obligations. So
this adapter is deliberately conservative:

- **Boilerplate strip (C3):** if ``trafilatura`` (the optional ``[url]`` extra) is
  installed it extracts the main content — dropping nav/ads/footers — and best-effort
  publication date. Without it, we fall back to the stdlib HTML strip (``normalize``):
  still usable, just noisier. So the dependency is genuinely optional.
- **robots.txt:** honored by default (stdlib ``urllib.robotparser``), fetched once per
  host and cached; a disallowed URL is skipped with a warning, not fetched.
- **Rate limit:** a configurable delay between page fetches, so a batch doesn't hammer
  a host.

Network I/O goes through ``_get`` so tests can drive the whole path offline.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from horizon.ingest.base import RawItem, Source
from horizon.ingest.normalize import normalize_text

# A descriptive, honest User-Agent so hosts (and their robots.txt) can identify us.
USER_AGENT = "HorizonBot/0.1 (+horizon-scanning; corpus gathering)"


def _domain(url: str) -> str:
    """The host part of ``url``, used as the ingest ``source`` label (else ``"url"``)."""
    return urlsplit(url).netloc or "url"


class UrlSource(Source):
    """Fetch a list of URLs and yield one :class:`RawItem` per readable page.

    ``urls`` is a single URL or an iterable of them. ``min_delay`` seconds pass between
    successive page fetches (rate limit). ``obey_robots`` (default True) skips URLs a
    host's robots.txt disallows for our User-Agent. ``timeout`` bounds each request.
    """

    def __init__(
        self,
        urls: str | Iterable[str],
        *,
        min_delay: float = 1.0,
        obey_robots: bool = True,
        timeout: float = 20.0,
        user_agent: str = USER_AGENT,
    ) -> None:
        """Store the URL list and fetch policy (delay, robots, timeout)."""
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.min_delay = min_delay
        self.obey_robots = obey_robots
        self.timeout = timeout
        self.user_agent = user_agent
        self._robots: dict[str, RobotFileParser | None] = {}

    def _get(self, url: str) -> str | None:
        """Fetch ``url`` as decoded text, or None on any network/decoding failure."""
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="replace")
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _allowed(self, url: str) -> bool:
        """Whether robots.txt permits us to fetch ``url`` (True when robots is off/absent)."""
        if not self.obey_robots:
            return True
        parts = urlsplit(url)
        host = parts.netloc
        if host not in self._robots:
            txt = self._get(f"{parts.scheme}://{host}/robots.txt")
            if txt is None:
                # Unreachable robots.txt -> don't block (common crawler convention).
                self._robots[host] = None
            else:
                rp = RobotFileParser()
                rp.parse(txt.splitlines())
                self._robots[host] = rp
        rp = self._robots[host]
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    def _extract(self, html: str) -> tuple[str, str | None, str | None, dict]:
        """Return ``(text, published_at, lang, meta)`` — trafilatura if present, else raw.

        The trafilatura path drops boilerplate, pulls a best-effort date/language, and
        harvests the rest of the page metadata (title, author, sitename, description,
        categories, tags, …) into ``meta``; the fallback returns the raw HTML for
        ``normalize_text`` to strip minimally and an empty ``meta``.
        """
        try:
            import trafilatura
        except ImportError:
            return html, None, None, {}  # minimal path: normalize_text strips the tags
        text = (
            trafilatura.extract(html, include_comments=False, include_tables=False)
            or ""
        )
        date = lang = None
        meta: dict = {}
        try:
            md = trafilatura.extract_metadata(html)
            if md is not None:
                date = getattr(md, "date", None)
                lang = getattr(md, "language", None)
                for key in (
                    "title",
                    "author",
                    "sitename",
                    "description",
                    "categories",
                    "tags",
                    "license",
                    "hostname",
                    "fingerprint",
                ):
                    val = getattr(md, key, None)
                    if val:
                        meta[key] = val
        except Exception:  # metadata is best-effort; never fail a fetch over it
            pass
        return text, date, lang, meta

    def fetch(self) -> Iterable[RawItem]:
        """Fetch each URL and yield a :class:`RawItem` per readable page.

        Robots-disallowed, unfetchable, and empty-after-normalization pages are skipped
        (with a warning for the first two); ``min_delay`` seconds pass between fetches.
        The full HTML is kept as ``raw`` for later re-normalization.
        """
        items: list[RawItem] = []
        for i, url in enumerate(self.urls):
            if not self._allowed(url):
                print(f"warning: robots.txt disallows {url}; skipped", file=sys.stderr)
                continue
            html = self._get(url)
            if not html:
                print(f"warning: could not fetch {url}; skipped", file=sys.stderr)
                continue
            extracted, published_at, lang, meta = self._extract(html)
            text = normalize_text(extracted)
            if not text:
                continue
            items.append(
                RawItem(
                    text=text,
                    source=_domain(url),
                    url=url,
                    published_at=published_at,
                    lang=lang,
                    meta=meta,
                    raw=html,  # full fetched HTML kept for later re-normalization
                    raw_suffix=".html",
                )
            )
            # Rate limit between page fetches (not after the last one).
            if self.min_delay and i < len(self.urls) - 1:
                time.sleep(self.min_delay)
        return items
