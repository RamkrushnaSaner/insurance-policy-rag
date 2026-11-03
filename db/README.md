# Database schema (Postgres + pgvector)

The schema the pgvector search and the table route read (`retrieval.dense_backend: pgvector`, `tables.enabled`).
The files hold the schema only: no policy text, no connection details.

| File | sha256 | What it adds |
|---|---|---|
| [`migrations/001_initial_schema.sql`](migrations/001_initial_schema.sql) | `c6d3ff4b…66cc` | Schema `rag`: documents and versions, text blocks, text chunks, embedding models and vectors, tables as printed (cells, references, links to chunks), the typed tables of migration 001, and the groups `rag_reader` and `rag_writer` |
| [`migrations/002_typed_tables.sql`](migrations/002_typed_tables.sql) | `545c9abe…6b15` | A typed table for every other printed table: the lists (with keyword indexes), claim deadlines, grievance contacts, the worked examples |

Each migration runs once, in one transaction, as the database owner, and must match its checksum. The development
database on Neon was migrated with the checked runner of the private ingestion bundle, which rehearses a migration,
records its checksum in `rag.schema_migrations` and verifies the permissions. For a fresh test database, `psql` is
enough (this is what CI does):

```bash
psql "$OWNER_URL" -v ON_ERROR_STOP=1 --single-transaction -f db/migrations/001_initial_schema.sql
psql "$OWNER_URL" -v ON_ERROR_STOP=1 --single-transaction -f db/migrations/002_typed_tables.sql
```

**Logins.** The groups have no login. The application connects as a login that is only a member of `rag_reader`,
with read-only transactions and a statement timeout set on the role, so a pooled connection keeps them too. Loading
a policy uses a short-lived login in `rag_writer`, closed after the load. Connection strings live in environment
variables (`DATABASE_URL`), never in files.
