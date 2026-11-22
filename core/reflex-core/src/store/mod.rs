//! Local SQLite storage. Every keystroke is kept forever; sessions older than
//! [`ARCHIVE_AFTER_DAYS`] are moved into zstd-compressed archive rows.

use crate::content::{ContentKind, Lang};
use crate::layout::{Geometry, LayoutId};
use crate::record::{CharRecord, TypedLine};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ARCHIVE_AFTER_DAYS: i64 = 30;
const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("archive error: {0}")]
    Archive(String),
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
        tx.commit()?;
        Ok(line_id)
    }

    pub fn insert_batch(&self, b: &BatchRecord) -> Result<()> {
        self.conn.execute(
            "INSERT INTO batches (session_id, ended_ms, seconds, chars, errors, wpm, accuracy, real_wpm, arm)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![b.session_id, b.ended_ms, b.seconds, b.chars, b.errors, b.wpm, b.accuracy, b.real_wpm, b.arm],
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
            "SELECT session_id, ended_ms, seconds, chars, errors, wpm, accuracy, real_wpm, arm
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
                    .prepare("SELECT pos, t_us, wall_ms, typed, key, modifiers, correct FROM keys WHERE line_id = ?1")?
                    .query_map([id], |r| {
                        Ok(KeyEventRecord {
                            pos: r.get(0)?,
                            t_us: r.get(1)?,
                            wall_ms: r.get(2)?,
                            typed: r.get(3)?,
                            key: r.get(4)?,
                            modifiers: r.get(5)?,
                            correct: r.get(6)?,
                        })
                    })?
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

    /// Export everything as JSON (lines with raw keys, batches).
    pub fn export_json(&self) -> Result<String> {
        let lines = self.typed_lines()?;
        let batches = self.batches()?;
        serde_json::to_string_pretty(&serde_json::json!({ "lines": lines, "batches": batches }))
            .map_err(|e| StoreError::Archive(e.to_string()))
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
}
