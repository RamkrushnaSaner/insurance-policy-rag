import json

import pytest

from policy_rag.corpus import load_corpus

from conftest import DATA

SAMPLE = {"id": "sample", "name": "Sample Health Shield", "chunks": str(DATA / "sample_policy_chunks.json")}
SECOND = {"id": "second", "name": "Sample Family Care", "chunks": str(DATA / "second_policy_chunks.json")}


def test_single_document_uses_labels_as_given():
    c = load_corpus([SAMPLE])
    assert len(c.chunks) == 15 and not c.multi_document
    assert c.chunks[3].label == "Section 3.1 - Hospitalization Expenses"
    assert c.chunks[3].embed_text == "Section 3.1 - Hospitalization Expenses: " + c.chunks[3].text
    assert sum(ch.index == "premium" for ch in c.chunks) == 2
    assert c.fingerprint == c.documents["sample"].sha256


def test_several_documents_get_distinct_ids_and_prefixed_labels():
    c = load_corpus([SAMPLE, SECOND])
    assert c.multi_document and len(c.chunks) == 18
    assert [ch.idx for ch in c.chunks] == list(range(18))
    second = [ch for ch in c.chunks if ch.doc_id == "second"]
    assert second[0].label.startswith("Sample Family Care | Section 0")
    assert ("sample", "3.1") in c.sections and ("second", "3.1") in c.sections     # same section id, two policies
    assert c.sections[("second", "3.1")].chunk_ids == [16]
    assert len(c.doc_chunk_ids(["second"])) == 3
    with pytest.raises(KeyError):
        c.doc_chunk_ids(["missing"])


def test_minimal_chunk_file_and_csv(tmp_path):
    p = tmp_path / "min.json"
    p.write_text(json.dumps({"chunks": [{"section_id": "1", "text": "Hello."}, {"section_id": "", "text": "skipped"}]}))
    c = load_corpus([{"id": "m", "chunks": str(p)}])
    assert c.chunks[0].label == "Section 1" and c.chunks[0].index == "policy" and c.documents["m"].skipped_rows == 1
    q = tmp_path / "min.csv"
    q.write_text("section_id,section_title,text\n2,Claims,Notify us.\n")
    c = load_corpus([{"id": "m", "chunks": str(q)}])
    assert c.chunks[0].label == "Section 2 - Claims"
