-- Database met fictieve longkankerpatienten op basis van IKNL
DROP TABLE IF EXISTS patient;

CREATE TABLE patient (
  id                  INTEGER PRIMARY KEY,
  display_name        TEXT NOT NULL,
  leeft               INTEGER, -- Leeftijd op incidentiedatum
  gesl                INTEGER, -- Geslacht (1 = man, 2 = vrouw)
  cstadium            TEXT,    -- Klinisch stadium (bijv. '1', '2', '3A', '4')
  tumsoort            INTEGER, -- Tumorsoort code (bijv. 302310)
  org_chir            INTEGER, -- Orgaanchirurgie (0 = nee, 1 = ja)
  chemo               INTEGER, -- Chemotherapie (0 = nee, 1 = ja)
  vit_stat            INTEGER, -- Vitale status (0 = in leven, 1 = overleden)
  vit_stat_int        INTEGER  -- Interval incidentiedatum tot vitale status (dagen)
);

INSERT INTO patient
  (display_name, leeft, gesl, cstadium, tumsoort, org_chir, chemo, vit_stat, vit_stat_int)
VALUES
  ('Patiënt A (Stadium I)', 62, 2, '1', 302310, 1, 0, 0, 1825),
  ('Patiënt B (Stadium III)', 58, 1, '3A', 302310, 0, 1, 1, 390),
  ('Patiënt C (Stadium IV)', 71, 1, '4', 302340, 0, 1, 1, 180);