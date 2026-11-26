-- Migration 004: the printed age of each premium row also as numbers (v3.6). Run once by the database owner, in ONE
-- transaction, after 003. Contains no policy text. No data is changed or reloaded.
--
-- Why: the printed age is text with marks ('43', '88*', '>=90*'), so an age compared as a number found no row
-- ('85' is printed '85*'), and a range of ages ('60 to 64', 'after 85') could not be written as a comparison.
-- What: policy_sql.premium_table gains age_from and age_to, read from the printed age by the view itself: '43' ->
-- 43, 43; '88*' -> 88, 88; '>=90*' -> 90, empty (90 and above). The printed age stays as it is, and the rows are
-- still chosen by it; the new columns are for comparing ages and for ranges.
-- CREATE OR REPLACE VIEW keeps the existing columns, in their order, and the grants; the new columns come last.

CREATE OR REPLACE VIEW policy_sql.premium_table AS
SELECT p.plan, p.tier, p.tier_cities, p.age, p.si_500000, p.si_1000000, p.si_1500000, p.si_2000000, p.si_2500000,
       p.si_5000000, p.si_10000000, p.si_20000000, t.pages[1] AS page,
       substring(p.age FROM '\d+')::integer                                          AS age_from,
       CASE WHEN p.age LIKE '>=%' THEN NULL ELSE substring(p.age FROM '\d+')::integer END AS age_to
  FROM rag.premium_table p
  JOIN rag.active_versions v ON v.version_id = p.version_id
  JOIN rag.tables t ON t.version_id = p.version_id AND t.table_id = p.table_id;

COMMENT ON COLUMN policy_sql.premium_table.age_from IS
  'First age of the printed age row, as a number (age 88* -> 88; >=90* -> 90): for comparing ages and for ranges';
COMMENT ON COLUMN policy_sql.premium_table.age_to IS
  'Last age of the printed age row, as a number (age 88* -> 88); NULL for an open row (>=90*: 90 and above)';
