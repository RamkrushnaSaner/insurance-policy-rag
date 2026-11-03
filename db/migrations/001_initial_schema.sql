-- Migration 001: initial schema for the policy RAG database (PostgreSQL 16+ with pgvector 0.7+; tested on 18.6 + 0.8.6).
-- Run once by the database owner, in ONE transaction, by the migration runner (it records this file's SHA-256 in
-- rag.schema_migrations). Contains no policy text.
--
-- Design
--   * Text is stored as chunks with vector embeddings. Tables are NOT embedded: they are stored as printed cells, with
--     the text that explains them (rag.table_refs), links to their text chunks (rag.chunk_tables), and typed fact
--     tables for exact SQL lookups and calculations.
--   * `record` (jsonb) keeps each extracted item exactly as the pipeline produced it. The query columns next to it are
--     GENERATED from it, so a column can never disagree with its record.
--   * Every typed fact row points to the printed cell it came from (table_id, row, col): that gives the page to cite.
--   * A policy is loaded as a new version ('loading'), checked, then made 'active'. One active version per document.
--   * Two permission groups (no login): rag_reader (read only) and rag_writer (load data). Login users are added to
--     them in later steps. Nothing is granted to PUBLIC.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA rag;
COMMENT ON SCHEMA rag IS 'Policy RAG data: documents, versions, text chunks + embeddings, tables + references + typed facts';

CREATE TABLE rag.schema_migrations (
  version     text PRIMARY KEY,                     -- '001'
  description text NOT NULL,
  sha256      text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  applied_at  timestamptz NOT NULL DEFAULT now(),
  applied_by  text NOT NULL DEFAULT current_user
);

-- jsonb array -> Postgres array (IMMUTABLE, so generated columns can use them)
CREATE FUNCTION rag.jsonb_int_array(j jsonb) RETURNS int[] LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
$$ SELECT coalesce(array_agg(x::int ORDER BY o), '{}') FROM jsonb_array_elements_text(j) WITH ORDINALITY AS t(x, o) $$;
CREATE FUNCTION rag.jsonb_text_array(j jsonb) RETURNS text[] LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
$$ SELECT coalesce(array_agg(x ORDER BY o), '{}') FROM jsonb_array_elements_text(j) WITH ORDINALITY AS t(x, o) $$;

-- ======================================================================================= documents and versions
CREATE TABLE rag.documents (
  doc_id   text PRIMARY KEY CHECK (doc_id ~ '^[a-z0-9][a-z0-9-]{1,62}$'),   -- 'optima-secure'
  title    text NOT NULL,
  insurer  text NOT NULL,
  uin      text,
  metadata jsonb NOT NULL                           -- identifiers, running headers, footer text, as extracted
);

CREATE TABLE rag.policy_versions (
  version_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  doc_id        text NOT NULL REFERENCES rag.documents,
  version_label text NOT NULL,
  source_file   text NOT NULL,
  source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
  pages         int  NOT NULL CHECK (pages > 0),
  pipeline      jsonb NOT NULL,                     -- extractor / chunker / linker versions, file hashes, model keys
  status        text NOT NULL DEFAULT 'loading' CHECK (status IN ('loading', 'active', 'retired')),
  loaded_at     timestamptz NOT NULL DEFAULT now(),
  activated_at  timestamptz,
  UNIQUE (doc_id, version_label),
  CHECK ((status = 'loading') = (activated_at IS NULL))
);
CREATE UNIQUE INDEX policy_versions_one_active ON rag.policy_versions (doc_id) WHERE status = 'active';
CREATE VIEW rag.active_versions AS SELECT doc_id, version_id, version_label, activated_at
                                     FROM rag.policy_versions WHERE status = 'active';

-- ======================================================================================= source text
CREATE TABLE rag.text_blocks (                      -- the extracted text pieces, in reading order
  version_id bigint NOT NULL REFERENCES rag.policy_versions ON DELETE CASCADE,
  block_id   text   NOT NULL,                       -- 'p7-b8'
  record     jsonb  NOT NULL,
  ord        int    GENERATED ALWAYS AS ((record->>'order')::int) STORED,
  page       int    GENERATED ALWAYS AS ((record->>'page')::int) STORED,
  type       text   GENERATED ALWAYS AS (record->>'type') STORED,
  text       text   GENERATED ALWAYS AS (record->>'text') STORED,
  PRIMARY KEY (version_id, block_id),
  CHECK (block_id = record->>'id')
);

-- ======================================================================================= text chunks (embedded)
CREATE TABLE rag.chunks (
  version_id    bigint NOT NULL REFERENCES rag.policy_versions ON DELETE CASCADE,
  chunk_uid     text   NOT NULL CHECK (chunk_uid ~ '^[0-9a-f]{32}$'),
                -- sha256(doc_id NUL embed_text), first 32 hex: same text -> same id in every version (vectors are reused)
  record        jsonb  NOT NULL,                    -- the chunk exactly as in the chunk file
  idx           int    GENERATED ALWAYS AS ((record->>'idx')::int) STORED,
  kind          text   GENERATED ALWAYS AS (record->>'kind') STORED,
  search_group  text   GENERATED ALWAYS AS (record->>'index') STORED,
  section_id    text   GENERATED ALWAYS AS (record->>'section_id') STORED,
  section_title text   GENERATED ALWAYS AS (record->>'section_title') STORED,
  clause        text   GENERATED ALWAYS AS (record->>'clause') STORED,
  label         text   GENERATED ALWAYS AS (record->>'label') STORED,
  group_key     text   GENERATED ALWAYS AS (record->>'group') STORED,
  pages         int[]  GENERATED ALWAYS AS (rag.jsonb_int_array(record->'pages')) STORED,
  block_ids     text[] GENERATED ALWAYS AS (rag.jsonb_text_array(record->'block_ids')) STORED,
  token_count   int    GENERATED ALWAYS AS ((record->>'token_count')::int) STORED,
  text          text   GENERATED ALWAYS AS (record->>'text') STORED,
  embed_text    text   GENERATED ALWAYS AS (record->>'embed_text') STORED,
  PRIMARY KEY (version_id, idx),
  UNIQUE (version_id, chunk_uid),
  CHECK (record->>'kind' = 'text'),                 -- tables are stored as tables, never as embedded chunks
  CHECK (record->>'index' IN ('policy', 'premium')),
  CHECK (jsonb_array_length(record->'pages') > 0 AND length(record->>'text') > 0)
);
CREATE INDEX chunks_filter ON rag.chunks (version_id, search_group, section_id);
CREATE INDEX chunks_uid    ON rag.chunks (chunk_uid);

-- ======================================================================================= embeddings
CREATE TABLE rag.embedding_models (
  model_key         text PRIMARY KEY,               -- model + pinned revision + dtype + dims
  model             text NOT NULL,
  revision          text NOT NULL,
  dim               int  NOT NULL CHECK (dim BETWEEN 1 AND 16000),
  dtype             text NOT NULL,
  pooling           text NOT NULL,
  query_instruction text,
  normalized        boolean NOT NULL,               -- unit length: inner product = cosine similarity
  notes             text,
  created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE rag.chunk_embeddings (
  chunk_uid         text   NOT NULL CHECK (chunk_uid ~ '^[0-9a-f]{32}$'),   -- matches rag.chunks.chunk_uid in any version
  model_key         text   NOT NULL REFERENCES rag.embedding_models,
  embed_text_sha256 text   NOT NULL CHECK (embed_text_sha256 ~ '^[0-9a-f]{64}$'),
  embedding         vector NOT NULL,                -- size set by the model (1,024 for Qwen3-Embedding-0.6B)
  created_at        timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (chunk_uid, model_key)
);
CREATE FUNCTION rag.check_embedding() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE m rag.embedding_models;
BEGIN
  SELECT * INTO m FROM rag.embedding_models WHERE model_key = NEW.model_key;
  IF vector_dims(NEW.embedding) <> m.dim THEN
    RAISE EXCEPTION 'embedding has % dimensions; model % expects %', vector_dims(NEW.embedding), NEW.model_key, m.dim;
  END IF;
  IF m.normalized AND abs(vector_norm(NEW.embedding) - 1) > 1e-3 THEN
    RAISE EXCEPTION 'embedding for % is not unit length (norm %)', NEW.chunk_uid, vector_norm(NEW.embedding);
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER chunk_embeddings_check BEFORE INSERT OR UPDATE ON rag.chunk_embeddings
  FOR EACH ROW EXECUTE FUNCTION rag.check_embedding();
-- No vector index while the corpus is small: every search compares all vectors (exact). Past ~50,000 chunks, add
-- one approximate index per model in a later migration, e.g.
--   CREATE INDEX ... ON rag.chunk_embeddings USING hnsw ((embedding::vector(1024)) vector_ip_ops) WHERE model_key = '...';

-- ======================================================================================= tables as printed
CREATE TABLE rag.tables (                           -- one row per printed table part (32 for this policy)
  version_id  bigint NOT NULL REFERENCES rag.policy_versions ON DELETE CASCADE,
  table_id    text   NOT NULL,                      -- 'p19-t1' = page 19, table 1
  record      jsonb  NOT NULL,                      -- caption, title, notes, bbox, printed grid, ...
  section_id  text   GENERATED ALWAYS AS (record->>'section_id') STORED,
  caption     text   GENERATED ALWAYS AS (record->>'caption') STORED,
  title       text   GENERATED ALWAYS AS (record->>'title') STORED,
  continues   text   GENERATED ALWAYS AS (record->>'continues') STORED,
  pages       int[]  GENERATED ALWAYS AS (rag.jsonb_int_array(record->'pages')) STORED,
  n_rows      int    GENERATED ALWAYS AS ((record->>'n_rows')::int) STORED,
  n_cols      int    GENERATED ALWAYS AS ((record->>'n_cols')::int) STORED,
  PRIMARY KEY (version_id, table_id),
  CHECK (table_id = record->>'table_id')
);

CREATE TABLE rag.table_cells (                      -- every printed cell
  version_id  bigint NOT NULL,
  table_id    text   NOT NULL,
  row         int    NOT NULL CHECK (row >= 0),
  col         int    NOT NULL CHECK (col >= 0),
  record      jsonb  NOT NULL,                      -- text, value, unit, spans, bbox, flags, header/row paths
  text        text    GENERATED ALWAYS AS (record->>'text') STORED,
  value       numeric GENERATED ALWAYS AS ((record->>'value')::numeric) STORED,     -- exact: 0.25 stays 0.25
  unit        text    GENERATED ALWAYS AS (record->>'unit') STORED,
  is_header   boolean GENERATED ALWAYS AS ((record->>'is_header')::boolean) STORED,
  header_path text[]  GENERATED ALWAYS AS (rag.jsonb_text_array(record->'header_path')) STORED,
  row_path    text[]  GENERATED ALWAYS AS (rag.jsonb_text_array(record->'row_path')) STORED,
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, coalesce(record->>'text', ''))) STORED,
  PRIMARY KEY (version_id, table_id, row, col),
  FOREIGN KEY (version_id, table_id) REFERENCES rag.tables ON DELETE CASCADE,
  CHECK (table_id = record->>'table_id' AND row = (record->>'row')::int AND col = (record->>'col')::int)
);
CREATE INDEX table_cells_tsv ON rag.table_cells USING gin (tsv);   -- keyword search inside tables ("hernia")

-- ======================================================================================= table references and links
CREATE TABLE rag.table_refs (                       -- one row per table (first printed part): the text that explains it
  version_id       bigint NOT NULL,
  table_id         text   NOT NULL,
  record           jsonb  NOT NULL,                 -- entry from policy_table_refs.json
  parts            text[] GENERATED ALWAYS AS (rag.jsonb_text_array(record->'parts')) STORED,
  section_id       text   GENERATED ALWAYS AS (record->>'section_id') STORED,
  clause           text   GENERATED ALWAYS AS (record->>'clause') STORED,
  caption          text   GENERATED ALWAYS AS (record->>'caption') STORED,
  title            text   GENERATED ALWAYS AS (record->>'title') STORED,
  reference_text   text   GENERATED ALWAYS AS (record->'reference'->>'text') STORED,
  reference_method text   GENERATED ALWAYS AS (record->'reference'->>'method') STORED,
  review_status    text   GENERATED ALWAYS AS (record->'review'->>'status') STORED,
  tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig,
                     coalesce(record->>'caption', '') || ' ' || coalesce(record->>'title', '') || ' ' ||
                     coalesce(record->'reference'->>'text', ''))) STORED,
  PRIMARY KEY (version_id, table_id),
  FOREIGN KEY (version_id, table_id) REFERENCES rag.tables ON DELETE CASCADE,
  CHECK (table_id = record->>'table_id' AND record->'parts'->>0 = table_id),
  CHECK (record->'reference'->>'method' IN ('pointer', 'assumptions', 'cross_ref', 'reviewed', 'heading')),
  CHECK (record->'review'->>'status' IN ('ok', 'heading-only'))   -- tables still needing review cannot be loaded
);
CREATE INDEX table_refs_tsv ON rag.table_refs USING gin (tsv);

CREATE TABLE rag.chunk_tables (                     -- which text chunks to show with a table, and why
  version_id bigint NOT NULL,
  table_id   text   NOT NULL,
  chunk_uid  text   NOT NULL,
  relation   text   NOT NULL CHECK (relation IN ('introduced_by', 'same_clause', 'notes_in', 'refers_to')),
  PRIMARY KEY (version_id, table_id, chunk_uid, relation),
  FOREIGN KEY (version_id, table_id)  REFERENCES rag.table_refs ON DELETE CASCADE,
  FOREIGN KEY (version_id, chunk_uid) REFERENCES rag.chunks (version_id, chunk_uid) ON DELETE CASCADE
);
CREATE INDEX chunk_tables_by_chunk ON rag.chunk_tables (version_id, chunk_uid);

-- ======================================================================================= typed fact tables
-- Built from the cells for the tables that questions filter or calculate on. Each row cites its cell.
CREATE TABLE rag.premium_rates (                    -- Annexure C
  version_id   bigint   NOT NULL,
  plan         text     NOT NULL,
  tier         smallint NOT NULL CHECK (tier IN (1, 2)),
  age_min      int      NOT NULL CHECK (age_min >= 0),
  age_max      int      CHECK (age_max IS NULL OR age_max >= age_min),   -- NULL: "and above"
  renewal_only boolean  NOT NULL,                   -- ages marked "*Only for Renewal Purposes"
  sum_insured  bigint   NOT NULL CHECK (sum_insured > 0),
  premium_inr  integer  NOT NULL CHECK (premium_inr > 0),                -- annual, excl. GST, whole rupees
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  UNIQUE (version_id, plan, tier, age_min, sum_insured),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.plan_sum_insured (                 -- Annexure A "Base Sum Insured": valid plan + Sum Insured pairs
  version_id  bigint NOT NULL,
  plan        text   NOT NULL,
  sum_insured bigint NOT NULL CHECK (sum_insured > 0),
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, plan, sum_insured),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.plan_benefits (                    -- Annexure A: one row per benefit per plan
  version_id  bigint NOT NULL,
  plan        text   NOT NULL,
  part        text   NOT NULL,                      -- 'Plan', 'Base Coverages', 'Optional Covers', 'Wellbeing Cover'
  benefit_no  text,
  benefit     text   NOT NULL,                      -- as printed
  printed     text   NOT NULL,                      -- the cell exactly as printed
  covered     boolean,
  scope       text,
  days        int    CHECK (days IS NULL OR days > 0),
  percent     numeric(6, 2) CHECK (percent IS NULL OR percent > 0),
  amount_inr  bigint CHECK (amount_inr IS NULL OR amount_inr > 0),
  per_day_inr int    CHECK (per_day_inr IS NULL OR per_day_inr > 0),
  max_inr     int    CHECK (max_inr IS NULL OR max_inr > 0),
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, plan, part, benefit),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.checkup_limits (                   -- Section 5
  version_id  bigint  NOT NULL,
  policy_type text    NOT NULL CHECK (policy_type IN ('individual', 'family_floater')),
  applies     text    NOT NULL,                     -- the printed sentence saying who the limit is for
  sum_insured bigint  NOT NULL CHECK (sum_insured > 0),
  si_printed  text    NOT NULL,
  limit_inr   integer NOT NULL CHECK (limit_inr > 0),
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, policy_type, sum_insured),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.deductible_discounts (             -- Section 8, Aggregate Deductible Discount
  version_id   bigint  NOT NULL,
  plan         text    NOT NULL,
  deductible   integer NOT NULL CHECK (deductible > 0),
  si_gt        bigint,                              -- Base SI must be > this (NULL: no lower bound)
  si_le        bigint,                              -- Base SI must be <= this (NULL: no upper bound)
  band_printed text,
  discount     numeric(5, 4) NOT NULL CHECK (discount > 0 AND discount < 1),
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  UNIQUE NULLS NOT DISTINCT (version_id, plan, deductible, si_gt, si_le),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.instalment_options (               -- Section 12.13
  version_id        bigint NOT NULL,
  option            text   NOT NULL,
  frequency         text   NOT NULL,
  grace_period_days int    NOT NULL CHECK (grace_period_days > 0),
  grace_printed     text   NOT NULL,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, option),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.claim_limit_illustration (         -- Section 2.1 worked example: a test case for the calculation code
  version_id   bigint NOT NULL,
  year         int    NOT NULL CHECK (year BETWEEN 1 AND 3),
  plan         text   NOT NULL,
  item         text   NOT NULL CHECK (item IN ('base_si', 'secure_benefit', 'bonus_or_plus', 'restore',
                                                 'max_single_claim', 'max_all_claims')),
  item_printed text   NOT NULL,
  amount_inr   bigint NOT NULL CHECK (amount_inr >= 0),   -- 'NIL' -> 0
  printed      text   NOT NULL,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, year, plan, item),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.city_tiers (                       -- Section 14: premium tier by city of residence (from the text)
  version_id      bigint   NOT NULL,
  place           text     NOT NULL,                -- as printed: 'Mumbai Suburban'; tier 2 row: 'Rest of India'
  place_key       text     GENERATED ALWAYS AS (lower(regexp_replace(place, '\s+', ' ', 'g'))) STORED,
  tier            smallint NOT NULL CHECK (tier IN (1, 2)),
  source_block_id text     NOT NULL,                -- the text block it was read from
  PRIMARY KEY (version_id, place),
  UNIQUE (version_id, place_key),
  FOREIGN KEY (version_id, source_block_id) REFERENCES rag.text_blocks (version_id, block_id) ON DELETE CASCADE
);

-- ======================================================================================= documentation in the catalog
COMMENT ON TABLE rag.chunks IS 'Text chunks (kind = text only). Embedded in rag.chunk_embeddings via chunk_uid.';
COMMENT ON TABLE rag.chunk_embeddings IS 'One vector per chunk text per model. Keyed by chunk_uid, shared across versions.';
COMMENT ON TABLE rag.tables IS 'Printed table parts as extracted. Not embedded.';
COMMENT ON TABLE rag.table_cells IS 'Every printed cell; tsv supports keyword search inside tables.';
COMMENT ON TABLE rag.table_refs IS 'Per table: the sentence(s) that introduce it, footnotes, review status.';
COMMENT ON TABLE rag.chunk_tables IS 'Links between tables and the text chunks shown with them.';

-- ======================================================================================= permissions
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_reader') THEN
    CREATE ROLE rag_reader NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_writer') THEN
    CREATE ROLE rag_writer NOLOGIN;
  END IF;
END $$;
REVOKE ALL ON SCHEMA rag FROM PUBLIC;
GRANT USAGE ON SCHEMA rag TO rag_reader, rag_writer;
-- readers: SELECT on everything except the migration log
GRANT SELECT ON ALL TABLES IN SCHEMA rag TO rag_reader;
REVOKE SELECT ON rag.schema_migrations FROM rag_reader;
-- writers: read + insert new data; may only change a version's status (no DELETE, no other UPDATE, no DDL)
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA rag TO rag_writer;
REVOKE INSERT ON rag.schema_migrations FROM rag_writer;
GRANT UPDATE (status, activated_at) ON rag.policy_versions TO rag_writer;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA rag TO rag_writer;
-- tables added by later migrations get the same rights
ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT SELECT ON TABLES TO rag_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT SELECT, INSERT ON TABLES TO rag_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT USAGE ON SEQUENCES TO rag_writer;
