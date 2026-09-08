CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_name TEXT NOT NULL,
    clock_in TEXT NOT NULL,
    clock_out TEXT,
    duration_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_name TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    camera_label TEXT,
    seen_at TEXT NOT NULL
);
