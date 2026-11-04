-- Settings that a person changes while the app runs, such as the selected model.
-- Secrets never go here: keys stay in the local .env file.
CREATE TABLE settings (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
