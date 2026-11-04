-- Orders. The status list matches RecordStatus in ai_ops/integrations/records.py.
-- The CHECK is a last line of defence, in case a bad value gets past the code.
CREATE TABLE records (
    id TEXT PRIMARY KEY,
    customer TEXT NOT NULL,
    item TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'shipped', 'delivered', 'cancelled'))
);
