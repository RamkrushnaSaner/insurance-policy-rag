"""Documents and their chunks: the input of the whole system.

A chunk file is a JSON list (or ``{"chunks": [...]}``) or a CSV with one row per chunk. Fields:

- ``text`` and ``section_id`` (required);
- ``section_title``, ``label`` (shown to the model above the passage; default ``"Section <id> - <title>"``),
  ``embed_text`` (what is searched; default ``"<label>: <text>"``), ``clause``;
- ``index``: ``"policy"`` (default) or the name of a separate index such as ``"premium"`` (see
  ``retrieval.routed_indexes`` in the config);
- ``group``: chunks that belong together (neighbour expansion stays inside a group; default: the section);
- ``kind`` / ``is_table``.

Any other field is ignored. Each document gets an id from the config, so several policies can be loaded at once.
Chunk ids (``Chunk.idx``) are positions in the combined corpus, in config order.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MAIN_INDEX = "policy"


@dataclass
class Chunk:
    idx: int
    doc_id: str
    section_id: str
    section_title: str
    clause: Optional[str]
    label: str
    index: str
    unit: str
    is_table: bool
    text: str
    embed_text: str

    @property
    def section_key(self) -> Tuple[str, str]:
        return (self.doc_id, self.section_id)

    @property
    def is_main(self) -> bool:
        return self.index == MAIN_INDEX


@dataclass
class Section:
    doc_id: str
    section_id: str
    title: str
    index: str
    chunk_ids: List[int] = field(default_factory=list)


@dataclass
class Document:
    id: str
    name: str
    path: str
    sha256: str
    n_chunks: int
    skipped_rows: int = 0


@dataclass
class Corpus:
    chunks: List[Chunk]
    documents: Dict[str, Document]
    sections: Dict[Tuple[str, str], Section]

    @property
    def multi_document(self) -> bool:
        return len(self.documents) > 1

    @property
    def fingerprint(self) -> str:
        """sha256 of the chunk file (one document), or of the documents' ids and file hashes (several)."""
        docs = list(self.documents.values())
        if len(docs) == 1:
            return docs[0].sha256
        return hashlib.sha256("\n".join(f"{d.id}:{d.sha256}" for d in docs).encode()).hexdigest()

    def section_text(self, key: Tuple[str, str]) -> str:
        return " ".join(self.chunks[i].text for i in self.sections[key].chunk_ids)

    def doc_chunk_ids(self, doc_ids: Optional[Iterable[str]]) -> Optional[set]:
        if not doc_ids:
            return None
        wanted = set(doc_ids)
        unknown = wanted - set(self.documents)
        if unknown:
            raise KeyError(f"unknown document id(s) {sorted(unknown)}; loaded: {sorted(self.documents)}")
        return {c.idx for c in self.chunks if c.doc_id in wanted}

    def summary(self) -> str:
        parts = []
        for d in self.documents.values():
            n_by_index: Dict[str, int] = {}
            for c in self.chunks:
                if c.doc_id == d.id:
                    n_by_index[c.index] = n_by_index.get(c.index, 0) + 1
            n_sec = sum(1 for k in self.sections if k[0] == d.id)
            idx = ", ".join(f"{n} {name}" for name, n in n_by_index.items())
            parts.append(f"{d.id}: {Path(d.path).name} (sha256 {d.sha256[:12]}…) {d.n_chunks} chunks ({idx}), "
                         f"{n_sec} sections" + (f"; {d.skipped_rows} rows skipped (no text or section_id)"
                                                if d.skipped_rows else ""))
        return "\n".join(parts)


def _clean(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()


def read_rows(path: Path) -> Tuple[List[dict], str]:
    """Rows of a chunk file and the file's sha256."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if Path(path).suffix.lower() == ".csv":
        import pandas as pd
        rows = pd.read_csv(io.BytesIO(raw), keep_default_na=False).to_dict("records")
    else:
        data = json.loads(raw.decode("utf-8"))
        rows = data.get("chunks", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of chunks")
    return rows, sha


def load_corpus(documents: Sequence) -> Corpus:
    """Load the chunk files of ``documents`` (``DocumentConfig`` objects or dicts with id, chunks, name).

    With more than one document, labels and search texts are prefixed with the document name so that the
    retriever and the answer model can tell the policies apart. With one document they are used as given."""
    docs = [SimpleNamespace(**d) if isinstance(d, dict) else d for d in documents]
    if not docs:
        raise ValueError("no documents configured: add at least one under 'documents' in the config")
    multi = len(docs) > 1
    chunks: List[Chunk] = []
    documents_out: Dict[str, Document] = {}
    sections: Dict[Tuple[str, str], Section] = {}
    for d in docs:
        path = Path(d.chunks)
        if not path.exists():
            raise FileNotFoundError(f"chunk file for document '{d.id}' not found: {d.chunks} (see data/README.md)")
        rows, sha = read_rows(path)
        name = getattr(d, "name", "") or d.id
        start, skipped = len(chunks), 0
        for r in rows:
            text, sid = _clean(r.get("text")), _clean(r.get("section_id"))
            if not text or not sid:
                skipped += 1
                continue
            title = _clean(r.get("section_title"))
            label = _clean(r.get("label")) or (f"Section {sid} - {title}" if title else f"Section {sid}")
            embed = _clean(r.get("embed_text")) or f"{label}: {text}"
            if multi:
                label, embed = f"{name} | {label}", f"{name} | {embed}"
            c = Chunk(idx=len(chunks), doc_id=d.id, section_id=sid, section_title=title,
                      clause=_clean(r.get("clause")) or None, label=label,
                      index=_clean(r.get("index")).lower() or MAIN_INDEX,
                      unit=_clean(r.get("group")) or sid,
                      is_table=str(r.get("is_table")).lower() == "true" or _clean(r.get("kind")) == "table",
                      text=text, embed_text=embed)
            chunks.append(c)
            s = sections.setdefault(c.section_key, Section(d.id, sid, title, c.index))
            s.chunk_ids.append(c.idx)
        if len(chunks) == start:
            raise ValueError(f"no usable chunks in {path.name}: each needs 'text' and 'section_id'")
        documents_out[d.id] = Document(d.id, name, str(path), sha, len(chunks) - start, skipped)
    return Corpus(chunks, documents_out, sections)
