//! The shared shape of typing data: real sessions and simulated typists both produce
//! [`TypedLine`]s, so training and evaluation never care where data came from.

use crate::content::{ContentKind, Lang};
use crate::layout::{Geometry, KeyId, LayoutId};
use serde::{Deserialize, Serialize};

/// First attempt at one character position.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CharRecord {
    pub pos: u16,
    /// Time since the previous correct character; `None` after a pause or at a start.
    pub latency_ms: Option<f32>,
    pub error: bool,
    /// Physical key hit instead of the right one, if known.
    pub wrong_key: Option<KeyId>,
    pub wall_ms: i64,
    /// Seconds of active typing in this session so far (fatigue signal).
    pub session_sec: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedLine {
    pub text: String,
    #[serde(with = "kind_serde")]
    pub kind: ContentKind,
    #[serde(with = "lang_serde")]
    pub lang: Lang,
    #[serde(with = "layout_serde")]
    pub layout: LayoutId,
    #[serde(with = "geometry_serde")]
    pub geometry: Geometry,
    /// Character typed right before this line (lines flow on with a space).
    pub prev_char: Option<char>,
    pub chars: Vec<CharRecord>,
}

impl TypedLine {
    pub fn started_ms(&self) -> i64 {
        self.chars.first().map_or(0, |c| c.wall_ms)
    }
}

macro_rules! str_serde {
    ($name:ident, $ty:ty) => {
        mod $name {
            use serde::{Deserialize, Deserializer, Serializer};
            pub fn serialize<S: Serializer>(v: &$ty, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(v.as_str())
            }
            pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<$ty, D::Error> {
                let s = String::deserialize(d)?;
                <$ty>::parse(&s).ok_or_else(|| serde::de::Error::custom(format!("unknown value {s}")))
            }
        }
    };
}

str_serde!(kind_serde, crate::content::ContentKind);
str_serde!(lang_serde, crate::content::Lang);
str_serde!(layout_serde, crate::layout::LayoutId);
str_serde!(geometry_serde, crate::layout::Geometry);
