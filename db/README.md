# Database schema (Postgres + pgvector)

The schema the pgvector search, the table route (v3.4) and the SQL step (v3.5) read (`retrieval.dense_backend:
pgvector`, `tables.enabled`, `sql.enabled`). The files hold the schema only: no policy text, no connection details.

| File | sha256 | What it adds |
|---|---|---|
| [`migrations/001_initial_schema.sql`](migrations/001_initial_schema.sql) | `c6d3ff4b…66cc` | Schema `rag`: documents and versions, text blocks, text chunks, embedding models and vectors, tables as printed (cells, references, links to chunks), the typed tables of migration 001, and the groups `rag_reader` and `rag_writer` |
| [`migrations/002_typed_tables.sql`](migrations/002_typed_tables.sql) | `545c9abe…6b15` | A typed table for every other printed table: the lists (with keyword indexes), claim deadlines, grievance contacts, the worked examples |
| [`migrations/003_numeric_tables_sql.sql`](migrations/003_numeric_tables_sql.sql) | `33189305…d3e5` | v3.5: drops the typed tables of 001 and 002; the three numeric tables (premiums, check-up limits, deductible discounts) as SQL tables in their printed layout; schema `policy_sql` (views of the active version, with the reference text and the printed column headings) and the group `rag_sql`, which reads `policy_sql` and nothing else; the link `read_with` |
| [`migrations/004_numeric_ages.sql`](migrations/004_numeric_ages.sql) | `94181d09…9d39` | v3.6: the printed age of each premium row also as numbers, `age_from` and `age_to` (`'88*'` → 88, 88; `'>=90*'` → 90, NULL), added by the view `policy_sql.premium_table` itself; no data is changed or reloaded |

Each migration runs once, in one transaction, as the database owner, and must match its checksum. The project's
Neon database is at 003 with v3.5's data since 16 November 2026; v3.4's data (at 002) is kept on a snapshot branch. The development
database on Neon was migrated with the checked runner of the private ingestion bundle, which rehearses a migration,
records its checksum in `rag.schema_migrations` and verifies the permissions. For a fresh test database, `psql` is
enough (this is what CI does):

```bash
psql "$OWNER_URL" -v ON_ERROR_STOP=1 --single-transaction -f db/migrations/001_initial_schema.sql
psql "$OWNER_URL" -v ON_ERROR_STOP=1 --single-transaction -f db/migrations/002_typed_tables.sql
psql "$OWNER_URL" -v ON_ERROR_STOP=1 --single-transaction -f db/migrations/003_numeric_tables_sql.sql   # v3.5
psql "$OWNER_URL" -v ON_ERROR_STOP=1 --single-transaction -f db/migrations/004_numeric_ages.sql         # v3.6
```

After 003, the v3.4 table route (`tables.enabled`) has no tables left to read: a database is either at 002 (v3.4) or
at 003 (v3.5). CI keeps one database of each. With 003 the views of `policy_sql` show the numeric tables of every
active version; they hold one policy for now.

**Logins.** The groups have no login. The application connects as a login that is only a member of `rag_reader`,
with read-only transactions and a statement timeout set on the role, so a pooled connection keeps them too. Loading
a policy uses a short-lived login in `rag_writer`, closed after the load. The SQL step (v3.5) runs the model's SQL as
a third login, only a member of `rag_sql`: it can read the views of `policy_sql` and nothing else, so even a query
that got past the checks could not read the chunks or write anything. For example (the owner runs this once; the
password goes into the environment variable `SQL_DATABASE_URL`, nowhere else):

```sql
CREATE ROLE rag_sql_app LOGIN PASSWORD '...' IN ROLE rag_sql;
ALTER ROLE rag_sql_app SET default_transaction_read_only = on;
ALTER ROLE rag_sql_app SET statement_timeout = '5s';
```

Connection strings live in environment variables (`DATABASE_URL`, `SQL_DATABASE_URL`), never in files.
