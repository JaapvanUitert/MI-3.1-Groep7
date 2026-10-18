-- Database met fictieve longkankerpatienten op basis van IKNL
DROP TABLE IF EXISTS patient;

CREATE TABLE patient (
  id                  INTEGER PRIMARY KEY,
  display_name        TEXT NOT NULL,
  leeft               INTEGER NOT NULL, -- Leeftijd in jaren
  gesl                INTEGER NOT NULL, -- 1 = Man, 2 = Vrouw
  stadium_num         INTEGER NOT NULL, -- 0 = In situ, 1 = Stadium I, 2 = Stadium II, 3 = Stadium III, 4 = Stadium IV
  SCLC                INTEGER NOT NULL, -- 0 = Niet-kleincellig (NSCLC), 1 = Kleincellig (SCLC)
  org_chir            INTEGER NOT NULL, -- Orgaanchirurgie: 0 = Nee, 1 = Ja
  chemo               INTEGER NOT NULL, -- Chemotherapie: 0 = Nee, 1 = Ja
  rt                  INTEGER NOT NULL, -- Radiotherapie: 0 = Nee, 1 = Ja
  imm                 INTEGER NOT NULL  -- Immunotherapie: 0 = Nee, 1 = Ja
);

INSERT INTO patient
  (display_name, leeft, gesl, stadium_num, SCLC, org_chir, chemo, rt, imm)
VALUES
  ('Patiënt 1 - Stadium I NSCLC (Vroeg stadium)',    62, 2, 1, 0, 1, 0, 0, 0),
  ('Patiënt 2 - Stadium III NSCLC (Lokaal uitgebreid)', 68, 1, 3, 0, 1, 1, 1, 1),
  ('Patiënt 3 - Stadium IV NSCLC (Gemetastaseerd)', 74, 1, 4, 0, 0, 1, 0, 1),
  ('Patiënt 4 - Stadium III SCLC (Kleincellig)', 59, 2, 3, 1, 0, 1, 1, 0),
  ('Patiënt 5 - Stadium II NSCLC (Resectabel)', 52, 1, 2, 0, 1, 1, 0, 0);