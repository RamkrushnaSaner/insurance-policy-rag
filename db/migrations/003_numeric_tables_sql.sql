-- Migration 003: numeric tables as SQL tables in their printed layout; every other table as text chunks (v3.5).
-- Run once by the database owner, in ONE transaction, after 001 and 002. Contains no policy text.
--
-- Design (replaces the typed tables of 001 and 002)
--   * Numeric tables stay tables, in their printed layout: one SQL row per printed row, one column per printed
--     column, the printed heading in each column's comment. Three tables: premiums (Annexure C), preventive health
--     check-up limits (Section 5) and aggregate deductible discounts (Section 8). Every row names its printed page
--     part and row; the printed text of every cell stays in rag.table_cells.
--   * Every other printed table is stored as text chunks with vectors (rag.chunks), like the rest of the policy.
--     Its typed table is dropped here, and its printed cells are no longer loaded.
--   * Reference text: each SQL table's printed title, the sentence that introduces it and its footnotes
--     (rag.tables, rag.table_refs), and the text chunks linked to it (rag.chunk_tables). The new link 'read_with'
--     names text an answer from the table must be read with (e.g. the Sum Insured options a plan offers).
--   * Schema policy_sql: what the answer model may query, for the active version only. The group rag_sql (no login;
--     logins are added later) can read policy_sql and nothing else: its views run with their owner's rights.

-- ======================================================================================= 1. typed tables: dropped
DROP TABLE rag.premium_rates, rag.plan_sum_insured, rag.plan_benefits, rag.checkup_limits, rag.deductible_discounts,
           rag.instalment_options, rag.claim_limit_illustration, rag.city_tiers,
           rag.eligible_relationships, rag.sum_insured_options, rag.major_medical_illnesses, rag.global_cover_benefits,
           rag.waiting_period_conditions, rag.non_payable_items, rag.claim_deadlines, rag.grievance_contacts,
           rag.si_utilisation_examples, rag.premium_examples, rag.premium_example_totals, rag.typed_loads;

-- ======================================================================================= 2. links: one more relation
ALTER TABLE rag.chunk_tables DROP CONSTRAINT chunk_tables_relation_check;
ALTER TABLE rag.chunk_tables ADD CONSTRAINT chunk_tables_relation_check
  CHECK (relation IN ('introduced_by', 'same_clause', 'notes_in', 'refers_to', 'read_with'));

-- ======================================================================================= 3. numeric tables, as printed
CREATE TABLE rag.premium_table (                    -- Annexure C: 4 printed tables (2 plans x 2 tiers), 91 rows each
  version_id  bigint   NOT NULL,
  plan        text     NOT NULL,                    -- from the printed title
  tier        smallint NOT NULL CHECK (tier IN (1, 2)),   -- from the printed title
  tier_cities text     NOT NULL,                    -- from the printed title: the places of the tier
  age         text     NOT NULL CHECK (age ~ '^(>=)?\d+\*?$'),   -- as printed: '43', '88*', '>=90*'
  si_500000   integer  NOT NULL CHECK (si_500000 > 0),
  si_1000000  integer  NOT NULL CHECK (si_1000000 > 0),
  si_1500000  integer  NOT NULL CHECK (si_1500000 > 0),
  si_2000000  integer  NOT NULL CHECK (si_2000000 > 0),
  si_2500000  integer  NOT NULL CHECK (si_2500000 > 0),
  si_5000000  integer  NOT NULL CHECK (si_5000000 > 0),
  si_10000000 integer  NOT NULL CHECK (si_10000000 > 0),
  si_20000000 integer  NOT NULL CHECK (si_20000000 > 0),
  table_id    text     NOT NULL,                    -- the printed page part ('p19-t1', or its continuation 'p20-t1')
  row         int      NOT NULL,                    -- the printed row in that part
  PRIMARY KEY (version_id, plan, tier, age),
  UNIQUE (version_id, table_id, row),
  FOREIGN KEY (version_id, table_id) REFERENCES rag.tables ON DELETE CASCADE
);

CREATE TABLE rag.checkup_limit_table (              -- Section 5: 2 printed tables, one printed row of limits each
  version_id              bigint  NOT NULL,
  policy_type             text    NOT NULL CHECK (policy_type IN ('individual', 'family_floater')),
  applies                 text    NOT NULL,         -- the printed sentence above the table
  limit_si_5_lakhs        integer NOT NULL CHECK (limit_si_5_lakhs > 0),
  limit_si_10_lakhs       integer NOT NULL CHECK (limit_si_10_lakhs > 0),
  limit_si_15_lakhs       integer NOT NULL CHECK (limit_si_15_lakhs > 0),
  limit_si_20_25_50_lakhs integer NOT NULL CHECK (limit_si_20_25_50_lakhs > 0),
  limit_si_100_200_lakhs  integer NOT NULL CHECK (limit_si_100_200_lakhs > 0),
  table_id                text    NOT NULL,
  row                     int     NOT NULL,         -- the printed "Limit of Cover" row
  PRIMARY KEY (version_id, policy_type),
  UNIQUE (version_id, table_id, row),
  FOREIGN KEY (version_id, table_id) REFERENCES rag.tables ON DELETE CASCADE
);

CREATE TABLE rag.deductible_discount_table (        -- Section 8: 1 printed table, 5 rows
  version_id                                bigint       NOT NULL,
  deductible_inr                            integer      NOT NULL CHECK (deductible_inr > 0),
  pct_suraksha_secure_super_secure_upto_20l numeric(5,2) NOT NULL CHECK (pct_suraksha_secure_super_secure_upto_20l > 0),
  pct_suraksha_secure_super_secure_over_20l numeric(5,2) NOT NULL CHECK (pct_suraksha_secure_super_secure_over_20l > 0),
  pct_secure_global                         numeric(5,2) NOT NULL CHECK (pct_secure_global > 0),
  pct_secure_global_plus                    numeric(5,2) NOT NULL CHECK (pct_secure_global_plus > 0),
  table_id                                  text         NOT NULL,
  row                                       int          NOT NULL,
  PRIMARY KEY (version_id, deductible_inr),
  UNIQUE (version_id, table_id, row),
  FOREIGN KEY (version_id, table_id) REFERENCES rag.tables ON DELETE CASCADE
);

-- ======================================================================================= 4. what the model may query
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_sql') THEN
    CREATE ROLE rag_sql NOLOGIN;
  END IF;
END $$;
CREATE SCHEMA policy_sql;
REVOKE ALL ON SCHEMA policy_sql FROM PUBLIC;

CREATE VIEW policy_sql.premium_table AS
SELECT p.plan, p.tier, p.tier_cities, p.age, p.si_500000, p.si_1000000, p.si_1500000, p.si_2000000, p.si_2500000,
       p.si_5000000, p.si_10000000, p.si_20000000, t.pages[1] AS page
  FROM rag.premium_table p
  JOIN rag.active_versions v ON v.version_id = p.version_id
  JOIN rag.tables t ON t.version_id = p.version_id AND t.table_id = p.table_id;

CREATE VIEW policy_sql.checkup_limit_table AS
SELECT c.policy_type, c.applies, c.limit_si_5_lakhs, c.limit_si_10_lakhs, c.limit_si_15_lakhs,
       c.limit_si_20_25_50_lakhs, c.limit_si_100_200_lakhs, t.pages[1] AS page
  FROM rag.checkup_limit_table c
  JOIN rag.active_versions v ON v.version_id = c.version_id
  JOIN rag.tables t ON t.version_id = c.version_id AND t.table_id = c.table_id;

CREATE VIEW policy_sql.deductible_discount_table AS
SELECT d.deductible_inr, d.pct_suraksha_secure_super_secure_upto_20l, d.pct_suraksha_secure_super_secure_over_20l,
       d.pct_secure_global, d.pct_secure_global_plus, t.pages[1] AS page
  FROM rag.deductible_discount_table d
  JOIN rag.active_versions v ON v.version_id = d.version_id
  JOIN rag.tables t ON t.version_id = d.version_id AND t.table_id = d.table_id;

-- the reference text: one row per printed table (a table printed over two pages is one row)
CREATE VIEW policy_sql.table_reference AS
WITH parts AS (
  SELECT DISTINCT 'premium_table' AS sql_table, version_id, table_id FROM rag.premium_table
  UNION SELECT DISTINCT 'checkup_limit_table', version_id, table_id FROM rag.checkup_limit_table
  UNION SELECT DISTINCT 'deductible_discount_table', version_id, table_id FROM rag.deductible_discount_table)
SELECT p.sql_table,
       r.table_id                                     AS printed_table,
       coalesce(r.title, r.caption, min(t.title))     AS printed_title,
       r.section_id,
       array_agg(DISTINCT t.pages[1] ORDER BY t.pages[1]) AS pages,
       r.reference_text                               AS introduced_by,
       (SELECT coalesce(jsonb_agg(f->>'text'), '[]') FROM jsonb_array_elements(r.record->'footnotes') f) AS footnotes,
       (SELECT coalesce(jsonb_agg(jsonb_build_object('relation', l.relation, 'label', c.label, 'text', c.text)
                                  ORDER BY c.idx, l.relation), '[]')
          FROM rag.chunk_tables l
          JOIN rag.chunks c ON c.version_id = l.version_id AND c.chunk_uid = l.chunk_uid
         WHERE l.version_id = r.version_id AND l.table_id = r.table_id) AS linked_text
  FROM parts p
  JOIN rag.active_versions v ON v.version_id = p.version_id
  JOIN rag.table_refs r ON r.version_id = p.version_id AND p.table_id = ANY(r.parts)
  JOIN rag.tables t ON t.version_id = p.version_id AND t.table_id = p.table_id
 GROUP BY p.sql_table, r.version_id, r.table_id, r.title, r.caption, r.section_id, r.reference_text, r.record;

-- every column of the three tables, with its type and printed heading (the column comments below)
CREATE VIEW policy_sql.column_reference AS
SELECT c.relname AS sql_table, a.attnum AS position, a.attname AS column_name,
       format_type(a.atttypid, a.atttypmod) AS data_type, col_description(a.attrelid, a.attnum) AS printed_heading
  FROM pg_attribute a
  JOIN pg_class c ON c.oid = a.attrelid
 WHERE c.relnamespace = 'policy_sql'::regnamespace
   AND c.relname IN ('premium_table', 'checkup_limit_table', 'deductible_discount_table')
   AND a.attnum > 0 AND NOT a.attisdropped;

COMMENT ON SCHEMA policy_sql IS 'The policy''s numeric tables in their printed layout, for model-written SQL (active version only).';
COMMENT ON VIEW policy_sql.premium_table IS 'Annexure C: gross annual premium (Rs., excl. GST) by age (rows) and Sum Insured (columns), one row per printed row of the 4 printed tables.';
COMMENT ON VIEW policy_sql.checkup_limit_table IS 'Section 5: Limit of Cover (Rs.) for the Preventive Health Check-up by Base Sum Insured under the Policy (columns).';
COMMENT ON VIEW policy_sql.deductible_discount_table IS 'Section 8: discount (%) on the Policy premium when Aggregate Deductible is opted for all Insured Persons.';
COMMENT ON VIEW policy_sql.table_reference IS 'Reference text of each printed numeric table: title, pages, introducing sentence, footnotes, linked text chunks.';
COMMENT ON VIEW policy_sql.column_reference IS 'Columns of the numeric tables with their types and printed headings.';
COMMENT ON COLUMN policy_sql.premium_table.plan IS 'Plan named in the printed title';
COMMENT ON COLUMN policy_sql.premium_table.tier IS 'Premium tier named in the printed title (1 or 2)';
COMMENT ON COLUMN policy_sql.premium_table.tier_cities IS 'Places of the tier, as printed in the title';
COMMENT ON COLUMN policy_sql.premium_table.age IS 'Printed heading: Age (as printed; * marks the printed footnote)';
COMMENT ON COLUMN policy_sql.premium_table.si_500000 IS 'Printed heading: Sum Insured 500,000 (premium in Rs.)';
COMMENT ON COLUMN policy_sql.premium_table.si_1000000 IS 'Printed heading: Sum Insured 1,000,000 (premium in Rs.)';
COMMENT ON COLUMN policy_sql.premium_table.si_1500000 IS 'Printed heading: Sum Insured 1,500,000 (premium in Rs.)';
COMMENT ON COLUMN policy_sql.premium_table.si_2000000 IS 'Printed heading: Sum Insured 2,000,000 (premium in Rs.)';
COMMENT ON COLUMN policy_sql.premium_table.si_2500000 IS 'Printed heading: Sum Insured 2,500,000 (premium in Rs.)';
COMMENT ON COLUMN policy_sql.premium_table.si_5000000 IS 'Printed heading: Sum Insured 5,000,000 (premium in Rs.)';
COMMENT ON COLUMN policy_sql.premium_table.si_10000000 IS 'Printed heading: Sum Insured 10,000,000 (premium in Rs.)';
COMMENT ON COLUMN policy_sql.premium_table.si_20000000 IS 'Printed heading: Sum Insured 20,000,000 (premium in Rs.)';
COMMENT ON COLUMN policy_sql.premium_table.page IS 'Page of the printed row';
COMMENT ON COLUMN policy_sql.checkup_limit_table.policy_type IS 'Individual or family floater: which printed table';
COMMENT ON COLUMN policy_sql.checkup_limit_table.applies IS 'The printed sentence above the table: who the limit is for';
COMMENT ON COLUMN policy_sql.checkup_limit_table.limit_si_5_lakhs IS 'Printed heading: 5 Lakhs (Limit of Cover in Rs.)';
COMMENT ON COLUMN policy_sql.checkup_limit_table.limit_si_10_lakhs IS 'Printed heading: 10 Lakhs (Limit of Cover in Rs.)';
COMMENT ON COLUMN policy_sql.checkup_limit_table.limit_si_15_lakhs IS 'Printed heading: 15 Lakhs (Limit of Cover in Rs.)';
COMMENT ON COLUMN policy_sql.checkup_limit_table.limit_si_20_25_50_lakhs IS 'Printed heading: 20, 25 & 50 Lakhs (Limit of Cover in Rs.)';
COMMENT ON COLUMN policy_sql.checkup_limit_table.limit_si_100_200_lakhs IS 'Printed heading: 100 & 200 Lakhs (Limit of Cover in Rs.)';
COMMENT ON COLUMN policy_sql.checkup_limit_table.page IS 'Page of the printed table';
COMMENT ON COLUMN policy_sql.deductible_discount_table.deductible_inr IS 'Printed heading: Deductible amount (INR)';
COMMENT ON COLUMN policy_sql.deductible_discount_table.pct_suraksha_secure_super_secure_upto_20l IS 'Printed heading: Optima Suraksha, Optima Secure & Optima Super Secure / Base SI <= 20 lakhs (discount %)';
COMMENT ON COLUMN policy_sql.deductible_discount_table.pct_suraksha_secure_super_secure_over_20l IS 'Printed heading: Optima Suraksha, Optima Secure & Optima Super Secure / Base SI > 20 lakhs (discount %)';
COMMENT ON COLUMN policy_sql.deductible_discount_table.pct_secure_global IS 'Printed heading: Optima Secure Global (discount %)';
COMMENT ON COLUMN policy_sql.deductible_discount_table.pct_secure_global_plus IS 'Printed heading: Optima Secure Global Plus (discount %)';
COMMENT ON COLUMN policy_sql.deductible_discount_table.page IS 'Page of the printed table';

-- ======================================================================================= 5. permissions
-- rag_writer loads the three tables (default privileges of 001 grant SELECT, INSERT; stated again here);
-- rag_reader (the application), rag_sql (model-written SQL) and rag_writer (to check a load) read policy_sql;
-- rag_sql reads nothing else.
GRANT SELECT, INSERT ON rag.premium_table, rag.checkup_limit_table, rag.deductible_discount_table TO rag_writer;
GRANT SELECT ON rag.premium_table, rag.checkup_limit_table, rag.deductible_discount_table TO rag_reader;
GRANT USAGE ON SCHEMA policy_sql TO rag_sql, rag_reader, rag_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA policy_sql TO rag_sql, rag_reader, rag_writer;
