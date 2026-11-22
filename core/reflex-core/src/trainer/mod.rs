//! Trainers decide what to type next. The ML planner is the real one; the others are
//! baselines used as fallbacks and as yardsticks in the benchmarks.

pub mod lines;
pub mod random;

use crate::content::Pool;
use crate::layout::Layout;
use crate::session::PlannedLine;
use rand::rngs::StdRng;

/// Everything a trainer may look at when choosing the next line.
pub struct PlanContext<'a> {
    pub pool: &'a Pool,
    pub layout: &'a Layout,
    pub recent: &'a lines::RecentWords,
}

pub trait LineSource: Send {
    fn next_line(&mut self, ctx: &PlanContext, rng: &mut StdRng) -> PlannedLine;
}
