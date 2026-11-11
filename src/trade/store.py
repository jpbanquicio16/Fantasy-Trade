"""SQLite log: recommendations, proposals you actually sent, and the responses.

The proposals/responses tables are how acceptance data begin to exist (design sections 7, 9, 12:
"log every proposal and response from day one"). Completed trades on Sleeper are not acceptance
data; these rows are.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from trade import __version__
from trade.config import Paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations(
  rec_id TEXT PRIMARY KEY, created_at TEXT, snapshot_id TEXT, config_hash TEXT, code_version TEXT, seed INTEGER,
  n_scenarios INTEGER, objective TEXT, offers TEXT, no_trade INTEGER, flags TEXT);
CREATE TABLE IF NOT EXISTS proposals(
  proposal_id INTEGER PRIMARY KEY AUTOINCREMENT, league_id TEXT, sent_at TEXT, from_roster INTEGER, to_roster INTEGER,
  give TEXT, get TEXT, rec_id TEXT, tier_shown TEXT, your_gain TEXT, partner_gain TEXT, market_change REAL, note TEXT);
CREATE TABLE IF NOT EXISTS responses(
  proposal_id INTEGER, outcome TEXT CHECK(outcome IN ('accepted','rejected','countered','expired','withdrawn')),
  responded_at TEXT, counter TEXT, note TEXT);
"""


def connect(paths: Paths | None = None) -> sqlite3.Connection:
    paths = (paths or Paths()).ensure()
    db = sqlite3.connect(paths.db)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def offer_json(rec, f) -> dict:
    ctx, o = rec.ctx, f.choice.offer
    nm = lambda p: ctx.snap.players[p].name if p in ctx.snap.players else p  # noqa: E731
    return {
        "offer_id": o.id, "partner": o.partner, "give": list(o.give), "get": list(o.get),
        "give_names": [nm(p) for p in o.give], "get_names": [nm(p) for p in o.get],
        "my_adds": f.choice.my.adds, "my_drops": f.choice.my.drops, "their_adds": f.choice.their.adds,
        "their_drops": f.choice.their.drops, "tier": f.tier, "qualifies": f.qualifies, "why_not": f.why_not,
        "g": f.g.to_json() if f.g else None, "g_selection_stage": f.g_sel,
        "me": {k: v.to_json() for k, v in f.me.items()}, "them": {k: v.to_json() for k, v in f.them.items()},
        "market_change": f.market_change, "sensitivity": f.sens,
        "extra": {k: v for k, v in f.extra.items() if k != "starts"},
    }


def save_recommendation(rec, tcfg, paths: Paths | None = None) -> None:
    import hashlib

    ctx = rec.ctx
    with connect(paths) as db:
        db.execute("INSERT OR REPLACE INTO recommendations VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
            rec.rec_id, now(), ctx.snap.snapshot_id, hashlib.sha256(tcfg.model_dump_json().encode()).hexdigest()[:12],
            __version__, ctx.seed, rec.n_fresh, rec.objective,
            json.dumps([offer_json(rec, f) for f in rec.finalists], default=str), int(rec.no_trade),
            json.dumps(ctx.flags)))


def log_proposal(rec_id: str, offer_id: str, note: str = "", paths: Paths | None = None) -> int:
    with connect(paths) as db:
        row = db.execute("SELECT * FROM recommendations WHERE rec_id=?", (rec_id,)).fetchone()
        if row is None:
            raise SystemExit(f"unknown recommendation {rec_id}")
        offer = next((o for o in json.loads(row["offers"]) if o["offer_id"] == offer_id), None)
        if offer is None:
            raise SystemExit(f"offer {offer_id} is not in {rec_id}")
        cur = db.execute(
            "INSERT INTO proposals(league_id,sent_at,from_roster,to_roster,give,get,rec_id,tier_shown,your_gain,"
            "partner_gain,market_change,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("", now(), None, offer["partner"], json.dumps(offer["give_names"]), json.dumps(offer["get_names"]),
             rec_id, offer["tier"], json.dumps(offer["me"]), json.dumps(offer["them"]), offer["market_change"], note))
        return int(cur.lastrowid)


def log_response(proposal_id: int, outcome: str, note: str = "", counter: str | None = None,
                 paths: Paths | None = None) -> None:
    with connect(paths) as db:
        if db.execute("SELECT 1 FROM proposals WHERE proposal_id=?", (proposal_id,)).fetchone() is None:
            raise SystemExit(f"unknown proposal {proposal_id}")
        db.execute("INSERT INTO responses VALUES(?,?,?,?,?)", (proposal_id, outcome, now(), counter, note))


def tier_report(paths: Paths | None = None) -> list[dict]:
    """Observed acceptance per tier with its sample size (no probability model until ~100 responses)."""
    with connect(paths) as db:
        rows = db.execute(
            "SELECT p.tier_shown t, COUNT(*) n, SUM(r.outcome='accepted') acc FROM proposals p "
            "LEFT JOIN responses r USING(proposal_id) GROUP BY p.tier_shown").fetchall()
    return [dict(r) for r in rows]
