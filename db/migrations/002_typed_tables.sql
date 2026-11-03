-- Migration 002: a typed table for every printed table that 001 kept only as cells (lists, deadlines, contacts,
-- worked examples). Run once by the database owner, in ONE transaction, by the migration runner. No policy text.
--
-- Same rules as the typed tables of 001:
--   * each table has the printed table's real columns (not its printed layout: lists printed in two halves side by
--     side become one list), with types and checks;
--   * every row points to the printed cell it came from (table_id, row, col), so it can be traced and cited;
--     rag.table_cells stays the record of what was printed;
--   * full-text indexes on the item names (keyword search: "hernia", "attendant charges").

-- ======================================================================================= lists
CREATE TABLE rag.eligible_relationships (           -- Section 1: relationships that can be covered
  version_id   bigint NOT NULL,
  item_no      int    NOT NULL CHECK (item_no > 0),                       -- 'xi.' -> 11
  item_printed text   NOT NULL,
  relationship text   NOT NULL CHECK (relationship <> ''),
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, item_no),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.sum_insured_options (              -- Section 2: Sum Insured options of the product
  version_id  bigint NOT NULL,
  sum_insured bigint NOT NULL CHECK (sum_insured > 0),
  printed     text   NOT NULL,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, sum_insured),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.major_medical_illnesses (          -- Section 4.8 E-Opinion: Major Medical Illnesses
  version_id bigint NOT NULL,
  item_no    int    NOT NULL CHECK (item_no > 0),
  illness    text   NOT NULL CHECK (illness <> ''),
  tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, illness)) STORED,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, item_no),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);
CREATE INDEX major_medical_illnesses_tsv ON rag.major_medical_illnesses USING gin (tsv);

CREATE TABLE rag.global_cover_benefits (            -- Sections 4.9 and 4.10: covers available under each Global Health Cover
  version_id    bigint NOT NULL,
  cover_section text   NOT NULL,                    -- '4.9'
  cover         text   NOT NULL,                    -- 'Global Health Cover (Emergency Treatments Only)'
  benefit_code  text   NOT NULL CHECK (benefit_code ~ '^[A-Z]+ ?[0-9]+(\.[0-9]+)*$'),   -- 'B 3.1'
  base_section  text   GENERATED ALWAYS AS (regexp_replace(benefit_code, '^[A-Z]+ ?', '')) STORED,   -- '3.1'
  benefit       text   NOT NULL CHECK (benefit <> ''),
  tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, benefit)) STORED,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, cover_section, benefit_code),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);
CREATE INDEX global_cover_benefits_tsv ON rag.global_cover_benefits USING gin (tsv);

CREATE TABLE rag.waiting_period_conditions (        -- Section 10.1, Excl02: conditions with a specified waiting period
  version_id     bigint NOT NULL,
  exclusion_code text   NOT NULL CHECK (exclusion_code ~ '^Excl[0-9]{2}$'),
  category       text   NOT NULL CHECK (category IN ('illness', 'surgical_procedure')),
  list_no        int    NOT NULL CHECK (list_no > 0),                     -- reading order within the category
  condition      text   NOT NULL CHECK (condition <> ''),
  waiting_months int    NOT NULL CHECK (waiting_months > 0),            -- from the text block below (the table has none)
  waiting_block  text   NOT NULL,
  tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, condition)) STORED,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, category, list_no),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE,
  FOREIGN KEY (version_id, waiting_block) REFERENCES rag.text_blocks (version_id, block_id) ON DELETE CASCADE
);
CREATE INDEX waiting_period_conditions_tsv ON rag.waiting_period_conditions USING gin (tsv);

CREATE TABLE rag.non_payable_items (                -- Annexure B: items for which coverage is not available
  version_id bigint NOT NULL,
  list       text   NOT NULL,                       -- 'List I'
  list_title text   NOT NULL,
  item_no    int    NOT NULL CHECK (item_no > 0),
  item       text   NOT NULL CHECK (item <> ''),
  tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, item)) STORED,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, list, item_no),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);
CREATE INDEX non_payable_items_tsv ON rag.non_payable_items USING gin (tsv);

-- ======================================================================================= procedures and contacts
CREATE TABLE rag.claim_deadlines (                  -- Section 11.6: time limits to submit reimbursement claim documents
  version_id    bigint NOT NULL,
  claim_type    text   NOT NULL,
  deadline_days int    NOT NULL CHECK (deadline_days > 0),
  counted_from  text   NOT NULL,
  printed       text   NOT NULL,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,          -- the time-limit cell
  PRIMARY KEY (version_id, claim_type),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.grievance_contacts (               -- Section 12.18: where to raise a grievance, by escalation level
  version_id    bigint NOT NULL,
  contact_point text   NOT NULL,                    -- 'Contact us at', 'Write to us at', ...
  level_no      int    NOT NULL CHECK (level_no BETWEEN 1 AND 9),   -- 1 = first contact point, 2 = escalation level 1, ...
  level         text   NOT NULL,
  details       text   NOT NULL,                    -- as printed
  phones        text[] NOT NULL,
  emails        text[] NOT NULL,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, contact_point, level_no),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

-- ======================================================================================= worked examples
CREATE TABLE rag.si_utilisation_examples (          -- Section 13.2: how the Sum Insured is used, claim by claim
  version_id            bigint NOT NULL,
  example               int    NOT NULL CHECK (example > 0),
  claim_no              int    NOT NULL CHECK (claim_no > 0),
  claim_inr             bigint NOT NULL CHECK (claim_inr >= 0),
  base_available_inr    bigint CHECK (base_available_inr >= 0),       -- NULL: printed '-' (nothing left)
  plus_available_inr    bigint CHECK (plus_available_inr >= 0),
  secure_available_inr  bigint CHECK (secure_available_inr >= 0),
  restore_available_inr bigint CHECK (restore_available_inr >= 0),
  admissible_inr        bigint NOT NULL CHECK (admissible_inr >= 0),
  utilisation           text   NOT NULL,
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,          -- the row's first cell ('1st claim')
  PRIMARY KEY (version_id, example, claim_no),
  CHECK (admissible_inr <= claim_inr),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.premium_examples (                 -- Section 15: premium computation examples, one row per member and basis
  version_id        bigint NOT NULL,
  example           int    NOT NULL CHECK (example > 0),
  age               int    NOT NULL CHECK (age >= 0),
  basis             text   NOT NULL CHECK (basis IN ('individual_separate', 'individual_one_policy', 'family_floater')),
  premium_inr       int    NOT NULL CHECK (premium_inr > 0),
  discount_rate     numeric(4, 3) CHECK (discount_rate > 0 AND discount_rate < 1),   -- from the column heading
  discount_inr      int    CHECK (discount_inr >= 0),                   -- NULL: no discount column for this basis
  premium_after_inr int    CHECK (premium_after_inr > 0),
  sum_insured       bigint NOT NULL CHECK (sum_insured > 0),
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,          -- the row's first cell (the age)
  PRIMARY KEY (version_id, example, age, basis),
  CHECK ((discount_inr IS NULL) = (premium_after_inr IS NULL)),
  CHECK (premium_after_inr IS NULL OR premium_after_inr = premium_inr - discount_inr),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

CREATE TABLE rag.premium_example_totals (           -- Section 15: the printed total of each basis
  version_id bigint NOT NULL,
  example    int    NOT NULL CHECK (example > 0),
  basis      text   NOT NULL CHECK (basis IN ('individual_separate', 'individual_one_policy', 'family_floater')),
  total_inr  int    NOT NULL CHECK (total_inr > 0),
  table_id text NOT NULL, row int NOT NULL, col int NOT NULL,
  PRIMARY KEY (version_id, example, basis),
  FOREIGN KEY (version_id, table_id, row, col) REFERENCES rag.table_cells ON DELETE CASCADE
);

-- ======================================================================================= provenance
CREATE TABLE rag.typed_loads (                      -- typed rows added to a version after it was loaded
  load_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  version_id     bigint NOT NULL REFERENCES rag.policy_versions ON DELETE CASCADE,
  loader         text   NOT NULL,
  loader_version text   NOT NULL,
  rows           jsonb  NOT NULL,                   -- {"table": number of rows}
  loaded_at      timestamptz NOT NULL DEFAULT now(),
  loaded_by      text   NOT NULL DEFAULT current_user,
  UNIQUE (version_id, loader, loader_version)
);

COMMENT ON TABLE rag.waiting_period_conditions IS 'Excl02 conditions; waiting_months is read from the text block waiting_block (the table does not print it).';
COMMENT ON TABLE rag.premium_examples IS 'Section 15 examples; premium_after_inr = premium_inr - discount_inr (checked).';

-- ======================================================================================= permissions (as in 001)
GRANT SELECT ON rag.eligible_relationships, rag.sum_insured_options, rag.major_medical_illnesses,
                rag.global_cover_benefits, rag.waiting_period_conditions, rag.non_payable_items, rag.claim_deadlines,
                rag.grievance_contacts, rag.si_utilisation_examples, rag.premium_examples, rag.premium_example_totals,
                rag.typed_loads TO rag_reader;
GRANT SELECT, INSERT ON rag.eligible_relationships, rag.sum_insured_options, rag.major_medical_illnesses,
                rag.global_cover_benefits, rag.waiting_period_conditions, rag.non_payable_items, rag.claim_deadlines,
                rag.grievance_contacts, rag.si_utilisation_examples, rag.premium_examples, rag.premium_example_totals,
                rag.typed_loads TO rag_writer;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA rag TO rag_writer;
