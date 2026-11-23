//! Local SQLite storage. Every keystroke is kept forever; sessions older than
//! [`ARCHIVE_AFTER_DAYS`] are moved into zstd-compressed archive rows.

use crate::content::{ContentKind, Lang};
use crate::layout::{Geometry, LayoutId};
use crate::record::{CharRecord, TypedLine};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ARCHIVE_AFTER_DAYS: i64 = 30;
const SCHEMA_VERSION: i64 = 3;

const KEY_COLUMNS: &str = "session_id, text, kind, lang, layout, geometry, pos, t_us, wall_ms, typed, key, modifiers, correct";

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("archive error: {0}")]
    Archive(String),
    #[error("Reflex is already open in another window or process using the same data")]
    Locked,
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// One raw key event, exactly as received from the UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyEventRecord {
    pub pos: u16,
    pub t_us: i64,
    pub wall_ms: i64,
    pub typed: String,
    pub key: Option<u8>,
    pub modifiers: u32,
    pub correct: bool,
}

/// Summary of one ~45 s practice batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchRecord {
    pub session_id: i64,
    pub ended_ms: i64,
    pub seconds: f64,
    pub chars: u32,
    pub errors: u32,
    pub wpm: f64,
    pub accuracy: f64,
    /// Model estimate of speed on natural text after this batch.
    pub real_wpm: Option<f64>,
    pub arm: Option<u32>,
    /// Local UTC offset when the batch ended, so it keeps its calendar day across
    /// daylight saving changes. `None` for batches stored before schema v3.
    #[serde(default)]
    pub utc_offset_min: Option<i32>,
}

impl BatchRecord {
    /// Local calendar day of this batch (its own offset, else `fallback_offset_min`).
    pub fn local_day(&self, fallback_offset_min: i32) -> i64 {
        crate::stats::day_index(self.ended_ms, self.utc_offset_min.unwrap_or(fallback_offset_min))
    }
}

pub struct Store {
    conn: Connection,
}

#[derive(Serialize, Deserialize)]
struct ArchivedLine {
    line: TypedLine,
    keys: Vec<KeyEventRecord>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Store> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let store = Store { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Store> {
        let store = Store { conn: Connection::open_in_memory()? };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        let version: i64 = self.conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if version < 1 {
            self.conn.execute_batch(
                "CREATE TABLE sessions (
                     id INTEGER PRIMARY KEY, started_ms INTEGER NOT NULL,
                     layout TEXT NOT NULL, geometry TEXT NOT NULL);
                 CREATE TABLE lines (
                     id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id),
                     text TEXT NOT NULL, kind TEXT NOT NULL, lang TEXT NOT NULL,
                     layout TEXT NOT NULL, geometry TEXT NOT NULL, prev_char TEXT);
                 CREATE TABLE chars (
                     line_id INTEGER NOT NULL REFERENCES lines(id), pos INTEGER NOT NULL,
                     latency_ms REAL, error INTEGER NOT NULL, wrong_key INTEGER,
                     wall_ms INTEGER NOT NULL, session_sec REAL NOT NULL,
                     PRIMARY KEY (line_id, pos)) WITHOUT ROWID;
                 CREATE TABLE keys (
                     line_id INTEGER NOT NULL REFERENCES lines(id), pos INTEGER NOT NULL,
                     t_us INTEGER NOT NULL, wall_ms INTEGER NOT NULL, typed TEXT NOT NULL,
                     key INTEGER, modifiers INTEGER NOT NULL, correct INTEGER NOT NULL);
                 CREATE INDEX keys_line ON keys(line_id);
                 CREATE TABLE batches (
                     id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, ended_ms INTEGER NOT NULL,
                     seconds REAL NOT NULL, chars INTEGER NOT NULL, errors INTEGER NOT NULL,
                     wpm REAL NOT NULL, accuracy REAL NOT NULL, real_wpm REAL, arm INTEGER);
                 CREATE TABLE archive (
                     session_id INTEGER PRIMARY KEY, started_ms INTEGER NOT NULL, blob BLOB NOT NULL);
                 CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
            )?;
        }
        if version < 2 {
            // Keys of the line being typed are saved as they arrive (open_keys) and move into
            // `keys` when the line is finished. Leftovers from a quit mid-line become
            // abandoned_keys: exported, but never used for training or stats.
            let columns = "seq INTEGER PRIMARY KEY, session_id INTEGER, text TEXT NOT NULL, kind TEXT NOT NULL,
                 lang TEXT NOT NULL, layout TEXT NOT NULL, geometry TEXT NOT NULL, pos INTEGER NOT NULL,
                 t_us INTEGER NOT NULL, wall_ms INTEGER NOT NULL, typed TEXT NOT NULL, key INTEGER,
                 modifiers INTEGER NOT NULL, correct INTEGER NOT NULL";
            self.conn.execute_batch(&format!(
                "CREATE TABLE open_keys ({columns}); CREATE TABLE abandoned_keys ({columns});"
            ))?;
        }
        if version < 3 {
            self.conn.execute_batch("ALTER TABLE batches ADD COLUMN utc_offset_min INTEGER;")?;
        }
        self.conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(())
    }

    pub fn begin_session(&self, layout: LayoutId, geometry: Geometry, now_ms: i64) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO sessions (started_ms, layout, geometry) VALUES (?1, ?2, ?3)",
            params![now_ms, layout.as_str(), geometry.as_str()],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Store a finished line with its derived per-character records and raw key events.
    pub fn insert_line(&mut self, session_id: i64, line: &TypedLine, keys: &[KeyEventRecord]) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO lines (session_id, text, kind, lang, layout, geometry, prev_char)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id,
                line.text,
                line.kind.as_str(),
                line.lang.as_str(),
                line.layout.as_str(),
                line.geometry.as_str(),
                line.prev_char.map(String::from),
            ],
        )?;
        let line_id = tx.last_insert_rowid();
        {
            let mut chars = tx.prepare_cached(
                "INSERT INTO chars (line_id, pos, latency_ms, error, wrong_key, wall_ms, session_sec)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for c in &line.chars {
                chars.execute(params![line_id, c.pos, c.latency_ms, c.error, c.wrong_key, c.wall_ms, c.session_sec])?;
            }
            let mut raw = tx.prepare_cached(
                "INSERT INTO keys (line_id, pos, t_us, wall_ms, typed, key, modifiers, correct)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for k in keys {
                raw.execute(params![line_id, k.pos, k.t_us, k.wall_ms, k.typed, k.key, k.modifiers, k.correct])?;
            }
        }
        tx.execute("DELETE FROM open_keys", [])?;
        tx.commit()?;
        Ok(line_id)
    }

    /// Save one key of the line being typed right away, so a quit mid-line loses nothing.
    pub fn insert_open_key(&self, session_id: Option<i64>, line: &OpenLine, k: &KeyEventRecord) -> Result<()> {
        self.conn
            .prepare_cached(&format!(
                "INSERT INTO open_keys ({KEY_COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)"
            ))?
            .execute(params![
                session_id,
                line.text,
                line.kind.as_str(),
                line.lang.as_str(),
                line.layout.as_str(),
                line.geometry.as_str(),
                k.pos,
                k.t_us,
                k.wall_ms,
                k.typed,
                k.key,
                k.modifiers,
                k.correct
            ])?;
        Ok(())
    }

    /// Keys of a line that was never finished (app quit or crashed mid-line, layout switched
    /// mid-line) are kept as abandoned keys. Returns how many were moved.
    pub fn abandon_open_keys(&mut self) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let n = tx.execute(&format!("INSERT INTO abandoned_keys ({KEY_COLUMNS}) SELECT {KEY_COLUMNS} FROM open_keys ORDER BY seq"), [])?;
        tx.execute("DELETE FROM open_keys", [])?;
        tx.commit()?;
        Ok(n)
    }

    pub fn insert_batch(&self, b: &BatchRecord) -> Result<()> {
        self.conn.execute(
            "INSERT INTO batches (session_id, ended_ms, seconds, chars, errors, wpm, accuracy, real_wpm, arm, utc_offset_min)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![b.session_id, b.ended_ms, b.seconds, b.chars, b.errors, b.wpm, b.accuracy, b.real_wpm, b.arm, b.utc_offset_min],
        )?;
        Ok(())
    }

    pub fn update_last_batch(&self, real_wpm: f64, arm: Option<u32>) -> Result<()> {
        self.conn.execute(
            "UPDATE batches SET real_wpm = ?1, arm = ?2 WHERE id = (SELECT MAX(id) FROM batches)",
            params![real_wpm, arm],
        )?;
        Ok(())
    }

    pub fn batches(&self) -> Result<Vec<BatchRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT session_id, ended_ms, seconds, chars, errors, wpm, accuracy, real_wpm, arm, utc_offset_min
             FROM batches ORDER BY ended_ms",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(BatchRecord {
                session_id: r.get(0)?,
                ended_ms: r.get(1)?,
                seconds: r.get(2)?,
                chars: r.get(3)?,
                errors: r.get(4)?,
                wpm: r.get(5)?,
                accuracy: r.get(6)?,
                real_wpm: r.get(7)?,
                arm: r.get(8)?,
                utc_offset_min: r.get(9)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// All typed lines in chronological order (archive included).
    pub fn typed_lines(&self) -> Result<Vec<TypedLine>> {
        let mut out = self.archived_lines()?;
        out.extend(self.live_lines(None)?.into_iter().map(|(_, l)| l));
        out.sort_by_key(TypedLine::started_ms);
        Ok(out)
    }

    /// Lines typed at or after `since_ms` (live tables only; recent data is never archived).
    pub fn typed_lines_since(&self, since_ms: i64) -> Result<Vec<TypedLine>> {
        Ok(self.live_lines(Some(since_ms))?.into_iter().map(|(_, l)| l).collect())
    }

    fn live_lines(&self, since_ms: Option<i64>) -> Result<Vec<(i64, TypedLine)>> {
        let mut stmt = self.conn.prepare(
            "SELECT l.id, l.text, l.kind, l.lang, l.layout, l.geometry, l.prev_char, l.session_id
             FROM lines l WHERE (?1 IS NULL OR
                 (SELECT MIN(wall_ms) FROM chars c WHERE c.line_id = l.id) >= ?1)
             ORDER BY l.id",
        )?;
        let heads: Vec<(i64, TypedLine)> = stmt
            .query_map(params![since_ms], |r| {
                let prev: Option<String> = r.get(6)?;
                Ok((
                    r.get::<_, i64>(0)?,
                    TypedLine {
                        text: r.get(1)?,
                        kind: ContentKind::parse(&r.get::<_, String>(2)?).unwrap_or(ContentKind::Words),
                        lang: Lang::parse(&r.get::<_, String>(3)?).unwrap_or(Lang::En),
                        layout: LayoutId::parse(&r.get::<_, String>(4)?).unwrap_or(LayoutId::DeIso),
                        geometry: Geometry::parse(&r.get::<_, String>(5)?).unwrap_or(Geometry::Iso),
                        prev_char: prev.and_then(|s| s.chars().next()),
                        chars: Vec::new(),
                    },
                ))
            })?
            .collect::<std::result::Result<_, _>>()?;
        let mut chars = self.conn.prepare_cached(
            "SELECT pos, latency_ms, error, wrong_key, wall_ms, session_sec FROM chars WHERE line_id = ?1 ORDER BY pos",
        )?;
        let mut out = Vec::with_capacity(heads.len());
        for (id, mut line) in heads {
            line.chars = chars
                .query_map([id], |r| {
                    Ok(CharRecord {
                        pos: r.get(0)?,
                        latency_ms: r.get(1)?,
                        error: r.get(2)?,
                        wrong_key: r.get(3)?,
                        wall_ms: r.get(4)?,
                        session_sec: r.get(5)?,
                    })
                })?
                .collect::<std::result::Result<_, _>>()?;
            if !line.chars.is_empty() {
                out.push((id, line));
            }
        }
        Ok(out)
    }

    fn archived_lines(&self) -> Result<Vec<TypedLine>> {
        let mut stmt = self.conn.prepare("SELECT blob FROM archive ORDER BY started_ms")?;
        let blobs: Vec<Vec<u8>> = stmt.query_map([], |r| r.get(0))?.collect::<std::result::Result<_, _>>()?;
        let mut out = Vec::new();
        for blob in blobs {
            let raw = zstd::decode_all(blob.as_slice()).map_err(|e| StoreError::Archive(e.to_string()))?;
            let lines: Vec<ArchivedLine> =
                serde_json::from_slice(&raw).map_err(|e| StoreError::Archive(e.to_string()))?;
            out.extend(lines.into_iter().map(|a| a.line));
        }
        Ok(out)
    }

    /// Compress sessions that started before `now - ARCHIVE_AFTER_DAYS` into the archive table.
    pub fn archive_old_sessions(&mut self, now_ms: i64) -> Result<usize> {
        let cutoff = now_ms - ARCHIVE_AFTER_DAYS * 86_400_000;
        let sessions: Vec<(i64, i64)> = self
            .conn
            .prepare("SELECT id, started_ms FROM sessions WHERE started_ms < ?1 AND id NOT IN (SELECT session_id FROM archive)")?
            .query_map([cutoff], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        let lines = self.live_lines(None)?;
        let tx = self.conn.transaction()?;
        for (session_id, started_ms) in &sessions {
            let ids: Vec<i64> = tx
                .prepare("SELECT id FROM lines WHERE session_id = ?1")?
                .query_map([session_id], |r| r.get(0))?
                .collect::<std::result::Result<_, _>>()?;
            let mut archived = Vec::new();
            for id in &ids {
                let Some((_, line)) = lines.iter().find(|(lid, _)| lid == id) else { continue };
                let keys: Vec<KeyEventRecord> = tx
                    .prepare("SELECT pos, t_us, wall_ms, typed, key, modifiers, correct FROM keys WHERE line_id = ?1 ORDER BY rowid")?
                    .query_map([id], key_record)?
                    .collect::<std::result::Result<_, _>>()?;
                archived.push(ArchivedLine { line: line.clone(), keys });
            }
            let json = serde_json::to_vec(&archived).map_err(|e| StoreError::Archive(e.to_string()))?;
            let blob = zstd::encode_all(json.as_slice(), 19).map_err(|e| StoreError::Archive(e.to_string()))?;
            tx.execute("INSERT INTO archive (session_id, started_ms, blob) VALUES (?1, ?2, ?3)", params![session_id, started_ms, blob])?;
            for id in &ids {
                tx.execute("DELETE FROM keys WHERE line_id = ?1", [id])?;
                tx.execute("DELETE FROM chars WHERE line_id = ?1", [id])?;
                tx.execute("DELETE FROM lines WHERE id = ?1", [id])?;
            }
        }
        tx.commit()?;
        Ok(sessions.len())
    }

    /// Most frequent mistakes in the live data: (expected char, key hit instead, count).
    pub fn confusions(&self, limit: usize) -> Result<Vec<(String, u8, u32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT substr(l.text, c.pos + 1, 1) AS expected, c.wrong_key, COUNT(*) AS n
             FROM chars c JOIN lines l ON l.id = c.line_id
             WHERE c.error = 1 AND c.wrong_key IS NOT NULL
             GROUP BY expected, c.wrong_key ORDER BY n DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Total characters typed (live + archived batches).
    pub fn total_chars(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COALESCE(SUM(chars), 0) FROM batches", [], |r| r.get(0))?)
    }

    pub fn setting(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Per-key numbers for the measured heatmap, for one layout (live data, last
    /// [`ARCHIVE_AFTER_DAYS`] days). Only first attempts at a character count.
    pub fn char_stats(&self, layout: LayoutId, geometry: Geometry) -> Result<Vec<CharStat>> {
        let mut stmt = self.conn.prepare(
            "SELECT substr(l.text, c.pos + 1, 1) AS expected, COUNT(*), SUM(c.error),
                    SUM(CASE WHEN c.error = 0 THEN c.latency_ms END),
                    SUM(CASE WHEN c.error = 0 AND c.latency_ms IS NOT NULL THEN 1 ELSE 0 END)
             FROM chars c JOIN lines l ON l.id = c.line_id
             WHERE l.layout = ?1 AND l.geometry = ?2
             GROUP BY expected",
        )?;
        let rows = stmt.query_map(params![layout.as_str(), geometry.as_str()], |r| {
            Ok(CharStat {
                expected: r.get::<_, String>(0)?.chars().next().unwrap_or(' '),
                attempts: r.get(1)?,
                errors: r.get(2)?,
                latency_sum_ms: r.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                latency_n: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Finished lines with their raw keys and session, oldest first (archive included).
    fn lines_with_keys(&self) -> Result<Vec<(i64, ArchivedLine)>> {
        let mut out = Vec::new();
        let mut stmt = self.conn.prepare("SELECT session_id, blob FROM archive ORDER BY started_ms")?;
        let blobs: Vec<(i64, Vec<u8>)> =
            stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<std::result::Result<_, _>>()?;
        for (session_id, blob) in blobs {
            let raw = zstd::decode_all(blob.as_slice()).map_err(|e| StoreError::Archive(e.to_string()))?;
            let lines: Vec<ArchivedLine> =
                serde_json::from_slice(&raw).map_err(|e| StoreError::Archive(e.to_string()))?;
            out.extend(lines.into_iter().map(|a| (session_id, a)));
        }
        let mut session = self.conn.prepare_cached("SELECT session_id FROM lines WHERE id = ?1")?;
        let mut keys = self.conn.prepare_cached(
            "SELECT pos, t_us, wall_ms, typed, key, modifiers, correct FROM keys WHERE line_id = ?1 ORDER BY rowid",
        )?;
        for (id, line) in self.live_lines(None)? {
            let session_id: i64 = session.query_row([id], |r| r.get(0))?;
            let keys = keys.query_map([id], key_record)?.collect::<std::result::Result<_, _>>()?;
            out.push((session_id, ArchivedLine { line, keys }));
        }
        out.sort_by_key(|(_, a)| a.line.started_ms());
        Ok(out)
    }

    /// Keys of lines that are not finished: from `table` (abandoned_keys or open_keys).
    fn unfinished_rows(&self, rows: &mut Vec<KeystrokeRow>, table: &str, status: KeyStatus) -> Result<()> {
        let mut stmt = self.conn.prepare(&format!("SELECT {KEY_COLUMNS} FROM {table} ORDER BY seq"))?;
        let keys: Vec<(Option<i64>, OpenLine, KeyEventRecord)> = stmt
            .query_map([], |r| {
                let line = OpenLine {
                    text: r.get(1)?,
                    kind: ContentKind::parse(&r.get::<_, String>(2)?).unwrap_or(ContentKind::Words),
                    lang: Lang::parse(&r.get::<_, String>(3)?).unwrap_or(Lang::En),
                    layout: LayoutId::parse(&r.get::<_, String>(4)?).unwrap_or(LayoutId::DeIso),
                    geometry: Geometry::parse(&r.get::<_, String>(5)?).unwrap_or(Geometry::Iso),
                };
                let key = KeyEventRecord {
                    pos: r.get(6)?,
                    t_us: r.get(7)?,
                    wall_ms: r.get(8)?,
                    typed: r.get(9)?,
                    key: r.get(10)?,
                    modifiers: r.get(11)?,
                    correct: r.get(12)?,
                };
                Ok((r.get(0)?, line, key))
            })?
            .collect::<std::result::Result<_, _>>()?;
        // Consecutive keys on the same text and layout, moving forward, belong to one line.
        let mut start = 0;
        while start < keys.len() {
            let same = |a: &OpenLine, b: &OpenLine| a.text == b.text && a.layout == b.layout && a.geometry == b.geometry;
            let mut end = start + 1;
            while end < keys.len() && same(&keys[end].1, &keys[start].1) && keys[end].2.pos >= keys[end - 1].2.pos {
                end += 1;
            }
            let line_keys: Vec<KeyEventRecord> = keys[start..end].iter().map(|(_, _, k)| k.clone()).collect();
            push_rows(rows, None, keys[start].0, status, &keys[start].1, &line_keys);
            start = end;
        }
        Ok(())
    }

    /// Every stored keystroke as one row: finished lines first (chronological), then keys
    /// of abandoned lines, then the line being typed right now.
    pub fn keystrokes(&self) -> Result<Vec<KeystrokeRow>> {
        let mut rows = Vec::new();
        for (line_no, (session_id, a)) in self.lines_with_keys()?.into_iter().enumerate() {
            let meta = OpenLine { text: a.line.text, kind: a.line.kind, lang: a.line.lang, layout: a.line.layout, geometry: a.line.geometry };
            push_rows(&mut rows, Some(line_no as u64 + 1), Some(session_id), KeyStatus::Finished, &meta, &a.keys);
        }
        self.unfinished_rows(&mut rows, "abandoned_keys", KeyStatus::Abandoned)?;
        self.unfinished_rows(&mut rows, "open_keys", KeyStatus::Open)?;
        Ok(rows)
    }

    /// All keystrokes as CSV (one row per key event, header included).
    pub fn export_csv(&self) -> Result<String> {
        let mut out = String::from(KeystrokeRow::CSV_HEADER);
        out.push('\n');
        for row in self.keystrokes()? {
            out.push_str(&row.csv_line());
            out.push('\n');
        }
        Ok(out)
    }

    /// Export everything as JSON: keystrokes (same rows as the CSV), finished lines with
    /// their per-character records, batches.
    pub fn export_json(&self) -> Result<String> {
        let lines = self.typed_lines()?;
        let batches = self.batches()?;
        let keystrokes = self.keystrokes()?;
        serde_json::to_string_pretty(&serde_json::json!({ "keystrokes": keystrokes, "lines": lines, "batches": batches }))
            .map_err(|e| StoreError::Archive(e.to_string()))
    }
}

fn key_record(r: &rusqlite::Row) -> rusqlite::Result<KeyEventRecord> {
    Ok(KeyEventRecord {
        pos: r.get(0)?,
        t_us: r.get(1)?,
        wall_ms: r.get(2)?,
        typed: r.get(3)?,
        key: r.get(4)?,
        modifiers: r.get(5)?,
        correct: r.get(6)?,
    })
}

fn push_rows(rows: &mut Vec<KeystrokeRow>, line_no: Option<u64>, session_id: Option<i64>, status: KeyStatus, line: &OpenLine, keys: &[KeyEventRecord]) {
    let chars: Vec<char> = line.text.chars().collect();
    let mut prev_us: Option<i64> = None;
    for k in keys {
        rows.push(KeystrokeRow {
            wall_ms: k.wall_ms,
            t_us: k.t_us,
            gap_ms: prev_us.map(|p| (k.t_us - p) as f64 / 1000.0),
            session_id,
            line_no,
            status,
            layout: line.layout.as_str().to_string(),
            geometry: line.geometry.as_str().to_string(),
            lang: line.lang.as_str().to_string(),
            kind: line.kind.as_str().to_string(),
            pos: k.pos,
            expected: chars.get(k.pos as usize).map(|c| c.to_string()).unwrap_or_default(),
            typed: k.typed.clone(),
            key_code: k.key.map(|key| crate::layout::KEY_CODES[key as usize].to_string()).unwrap_or_default(),
            modifiers: k.modifiers,
            correct: k.correct,
        });
        prev_us = Some(k.t_us);
    }
}

/// The line a key belongs to, before it is finished.
#[derive(Debug, Clone)]
pub struct OpenLine {
    pub text: String,
    pub kind: ContentKind,
    pub lang: Lang,
    pub layout: LayoutId,
    pub geometry: Geometry,
}

/// First-attempt numbers for one expected character.
#[derive(Debug, Clone, PartialEq)]
pub struct CharStat {
    pub expected: char,
    pub attempts: u32,
    pub errors: u32,
    pub latency_sum_ms: f64,
    pub latency_n: u32,
}

/// Whether a key belongs to a finished line. Only finished lines feed training and stats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyStatus {
    Finished,
    /// The line being typed right now.
    Open,
    /// A line never finished (quit or layout switch mid-line).
    Abandoned,
}

impl KeyStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            KeyStatus::Finished => "finished",
            KeyStatus::Open => "open",
            KeyStatus::Abandoned => "abandoned",
        }
    }
}

/// One exported key event.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KeystrokeRow {
    pub wall_ms: i64,
    /// Monotonic hardware timestamp (µs).
    pub t_us: i64,
    /// Time since the previous key of the same line (ms).
    pub gap_ms: Option<f64>,
    pub session_id: Option<i64>,
    /// Running number of the finished line; empty for unfinished lines.
    pub line_no: Option<u64>,
    pub status: KeyStatus,
    pub layout: String,
    pub geometry: String,
    pub lang: String,
    pub kind: String,
    pub pos: u16,
    pub expected: String,
    pub typed: String,
    /// Physical key by position (W3C code, e.g. "KeyA").
    pub key_code: String,
    pub modifiers: u32,
    pub correct: bool,
}

impl KeystrokeRow {
    pub const CSV_HEADER: &str =
        "wall_ms,t_us,gap_ms,session_id,line_no,status,layout,geometry,lang,kind,pos,expected,typed,key_code,modifiers,correct";

    fn csv_line(&self) -> String {
        let opt = |v: Option<String>| v.unwrap_or_default();
        [
            self.wall_ms.to_string(),
            self.t_us.to_string(),
            opt(self.gap_ms.map(|g| format!("{g:.3}"))),
            opt(self.session_id.map(|v| v.to_string())),
            opt(self.line_no.map(|v| v.to_string())),
            self.status.as_str().to_string(),
            csv_field(&self.layout),
            csv_field(&self.geometry),
            csv_field(&self.lang),
            csv_field(&self.kind),
            self.pos.to_string(),
            csv_field(&self.expected),
            csv_field(&self.typed),
            csv_field(&self.key_code),
            self.modifiers.to_string(),
            (self.correct as u8).to_string(),
        ]
        .join(",")
    }
}

/// Quote a CSV field when needed (RFC 4180). Spaces are quoted too, so a typed space
/// survives tools that trim fields.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r', ' ']) || s.is_empty() {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(ms: i64) -> TypedLine {
        TypedLine {
            text: "hallo welt".into(),
            kind: ContentKind::Words,
            lang: Lang::De,
            layout: LayoutId::DeIso,
            geometry: Geometry::Iso,
            prev_char: Some(' '),
            chars: (0..10)
                .map(|i| CharRecord {
                    pos: i,
                    latency_ms: Some(150.0),
                    error: i == 3,
                    wrong_key: (i == 3).then_some(5),
                    wall_ms: ms + i as i64 * 150,
                    session_sec: i as f32 * 0.15,
                })
                .collect(),
        }
    }

    #[test]
    fn roundtrip_and_archive() {
        let mut s = Store::open_in_memory().unwrap();
        let old = 1_000_000_000_000;
        let sid = s.begin_session(LayoutId::DeIso, Geometry::Iso, old).unwrap();
        s.insert_line(sid, &line(old), &[]).unwrap();
        let now = old + 40 * 86_400_000;
        let sid2 = s.begin_session(LayoutId::DeIso, Geometry::Iso, now).unwrap();
        s.insert_line(sid2, &line(now), &[]).unwrap();
        assert_eq!(s.typed_lines().unwrap().len(), 2);
        assert_eq!(s.archive_old_sessions(now).unwrap(), 1);
        let all = s.typed_lines().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0], line(old));
        assert_eq!(s.typed_lines_since(now).unwrap().len(), 1);
        s.set_setting("pool", "x").unwrap();
        assert_eq!(s.setting("pool").unwrap().as_deref(), Some("x"));
    }

    fn key(pos: u16, typed: &str, correct: bool, t_us: i64) -> KeyEventRecord {
        KeyEventRecord { pos, t_us, wall_ms: t_us / 1000, typed: typed.into(), key: Some(4), modifiers: 0, correct }
    }

    #[test]
    fn keystrokes_export_csv_and_abandoned_keys() {
        let mut s = Store::open_in_memory().unwrap();
        let t = 1_000_000_000_000;
        let sid = s.begin_session(LayoutId::DeIso, Geometry::Iso, t).unwrap();
        let open = OpenLine { text: "äb, ".into(), kind: ContentKind::Words, lang: Lang::De, layout: LayoutId::DeIso, geometry: Geometry::Iso };
        let keys = vec![key(0, "x", false, 1_000), key(0, "ä", true, 2_000), key(1, "b", true, 3_000)];
        for k in &keys {
            s.insert_open_key(Some(sid), &open, k).unwrap();
        }
        // The line finishes: its keys move to `keys`, open_keys is emptied.
        let mut finished = line(t);
        finished.text = open.text.clone();
        s.insert_line(sid, &finished, &keys).unwrap();
        // The next line is abandoned after one key (app quit).
        s.insert_open_key(Some(sid), &open, &key(0, ",", false, 9_000)).unwrap();
        assert_eq!(s.keystrokes().unwrap()[3].status, KeyStatus::Open);
        assert_eq!(s.abandon_open_keys().unwrap(), 1);
        assert_eq!(s.abandon_open_keys().unwrap(), 0);

        let rows = s.keystrokes().unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!((rows[0].expected.as_str(), rows[0].typed.as_str(), rows[0].correct), ("ä", "x", false));
        assert_eq!(rows[1].gap_ms, Some(1.0));
        assert!(rows[..3].iter().all(|r| r.line_no == Some(1) && r.status == KeyStatus::Finished));
        assert!(rows[3].status == KeyStatus::Abandoned && rows[3].line_no.is_none());
        // Abandoned keys never reach training data.
        assert_eq!(s.typed_lines().unwrap().len(), 1);

        let csv = s.export_csv().unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[0], KeystrokeRow::CSV_HEADER);
        assert!(lines[4].contains(",abandoned,") && lines[4].contains(",\",\",") && lines[4].ends_with(",0"), "{}", lines[4]);
        let json: serde_json::Value = serde_json::from_str(&s.export_json().unwrap()).unwrap();
        assert_eq!(json["keystrokes"].as_array().unwrap().len(), 4);

        // Keys survive archiving.
        s.archive_old_sessions(t + 40 * 86_400_000).unwrap();
        assert_eq!(s.keystrokes().unwrap(), rows);
    }

    #[test]
    fn char_stats_per_layout() {
        let mut s = Store::open_in_memory().unwrap();
        let sid = s.begin_session(LayoutId::DeIso, Geometry::Iso, 0).unwrap();
        s.insert_line(sid, &line(1_000), &[]).unwrap();
        let mut other = line(5_000);
        other.layout = LayoutId::UsAnsi;
        other.geometry = Geometry::Ansi;
        s.insert_line(sid, &other, &[]).unwrap();
        let stats = s.char_stats(LayoutId::DeIso, Geometry::Iso).unwrap();
        // "hallo welt": 'l' appears 3 times; position 3 (the second 'l') was an error.
        let l = stats.iter().find(|c| c.expected == 'l').unwrap();
        assert_eq!((l.attempts, l.errors, l.latency_n), (3, 1, 2));
        assert_eq!(stats.iter().map(|c| c.attempts).sum::<u32>(), 10);
    }
}
