"""Reddit closed-world authorship corpus loader.

Columns: ID, author, text, hs. ``author`` is a pseudonymous integer id, which is
exactly what we want: it supports an authorship-attribution attack without ever
touching a real identity. The corpus does double duty: ``hs`` gives the utility
(HSD macro-F1) signal, ``author`` gives the privacy (closed-world attribution)
signal, on a single corpus. N authors define the closed-world candidate set.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .base import DatasetLoader, Post


class RedditCorpus(DatasetLoader):
    name = "reddit_closed_world"

    def __init__(self, path: str | Path, min_chars: int = 1):
        self.path = Path(path)
        self.min_chars = min_chars

    def load(self) -> list[Post]:
        posts: list[Post] = []
        # csv field can exceed the default 128k limit on very long comments.
        csv.field_size_limit(10_000_000)
        with open(self.path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row["text"] or "").strip()
                if len(text) < self.min_chars:
                    continue
                posts.append(
                    Post(
                        id=row["ID"],
                        author=str(row["author"]),
                        text=text,
                        label=int(row["hs"]),
                    )
                )
        return posts

    @staticmethod
    def authors(posts: list[Post]) -> list[str]:
        return sorted({p.author for p in posts}, key=lambda a: (len(a), a))
