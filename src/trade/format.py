"""League format: standings, pairings, playoff bracket, trade deadline and trade timing.

The design (section 1) says the playoff format is configuration, never inference, and lists what
must be read: playoff teams, start week, round lengths, seeding, bracket shape, deadline, review
period. All of it comes from the Sleeper league object, the roster records, ``matchups/{week}``
and ``winners_bracket``. Verified for this league on 11 Nov 2026: 6 playoff teams from week 22,
one week per round, and a fixed bracket with byes for seeds 1 and 2 (seed 1 meets the winner of
4 v 5, seed 2 the winner of 3 v 6) - the design's "assume fixed bracket" default, now confirmed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

PT = ZoneInfo("America/Los_Angeles")


@dataclass
class Standing:
    wins: float = 0.0  # ties count half
    losses: float = 0.0
    pf: float = 0.0
    pa: float = 0.0


@dataclass(frozen=True)
class Match:
    """A playoff game. A source is ("seed", s) or ("win", match_id)."""

    m: int
    r: int  # round, 1-based
    t1: tuple[str, int]
    t2: tuple[str, int]


@dataclass
class LeagueFormat:
    roster_ids: list[int]
    playoff_teams: int
    first_unscored_week: int
    reg_last_week: int
    round_weeks: list[list[int]]  # weeks of each playoff round
    bracket: list[Match]  # dependency order; the last match is the final
    pairings: dict[int, list[tuple[int, int]]]  # week -> roster-id pairs
    standings: dict[int, Standing]
    trade_deadline_week: int | None
    review_days: int
    flags: list[str] = field(default_factory=list)
    bracket_source: str = "api"

    @property
    def n_teams(self) -> int:
        return len(self.roster_ids)

    @property
    def final_week(self) -> int:
        return self.round_weeks[-1][-1]

    @property
    def playoff_weeks(self) -> list[int]:
        return [w for ws in self.round_weeks for w in ws]

    def tix(self, rid: int) -> int:
        return self.roster_ids.index(rid)


# --- seeding -----------------------------------------------------------------------
def seed_key(wins: np.ndarray, pf: np.ndarray, pa: np.ndarray) -> np.ndarray:
    """Sort key, larger is better: record, then points for, then points against (Sleeper default;
    further ties - a coin toss - are broken by team index, which is arbitrary but deterministic)."""
    return wins * 1e7 + pf - pa * 1e-3


def rank_teams(wins, pf, pa) -> np.ndarray:
    """Team indices best-first along the last axis (stable, so exact ties go to the lower index)."""
    return np.argsort(-seed_key(wins, pf, pa), axis=-1, kind="stable")


# --- brackets ----------------------------------------------------------------------
def standard_bracket(k: int) -> list[Match]:
    """Fixed bracket: seed s meets seed (size + 1 - s); missing seeds are byes."""
    size = 1 << max(1, math.ceil(math.log2(max(k, 2))))
    order = [1]
    while len(order) < size:  # standard seeding order, e.g. 1 8 4 5 3 6 2 7
        n = len(order) * 2 + 1
        order = [x for s in order for x in (s, n - s)]
    matches: list[Match] = []
    nodes: list[tuple[str, int]] = [("seed", s) for s in order]
    m, r = 0, 1
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes), 2):
            a, b = nodes[i], nodes[i + 1]
            if a[0] == "seed" and a[1] > k or b[0] == "seed" and b[1] > k:  # bye: the real team advances
                nxt.append(b if a[0] == "seed" and a[1] > k else a)
                continue
            m += 1
            matches.append(Match(m, r, a, b))
            nxt.append(("win", m))
        nodes, r = nxt, r + 1
    return matches


def bracket_from_api(raw: list[dict], seed_of: dict[int, int], k: int) -> list[Match] | None:
    """Winners-bracket matches that feed the championship, in terms of seeds.

    Returns None when the bracket does not fit the standings (seeds in the first round must pair
    up as s + (k + 1 - s) = k + 1, or the bracket names a team outside the top k): the caller then
    falls back to ``standard_bracket`` and flags it.
    """
    by_m = {x["m"]: x for x in raw}
    final = next((x for x in raw if x.get("p") == 1), None)
    if final is None:
        return None

    def src(x: dict, side: str):
        frm = x.get(f"{side}_from")
        if frm and "w" in frm:
            return ("win", int(frm["w"]))
        rid = x.get(side)
        if rid is None or rid not in seed_of or seed_of[rid] > k:
            raise KeyError
        return ("seed", seed_of[rid])

    out: dict[int, Match] = {}

    def walk(m: int):
        if m in out:
            return
        x = by_m[m]
        t1, t2 = src(x, "t1"), src(x, "t2")
        for t in (t1, t2):
            if t[0] == "win":
                walk(t[1])
        out[m] = Match(m, int(x["r"]), t1, t2)

    try:
        walk(final["m"])
    except KeyError:
        return None
    first = [mt for mt in out.values() if mt.t1[0] == "seed" and mt.t2[0] == "seed" and mt.r == 1]
    size = 1 << max(1, math.ceil(math.log2(max(k, 2))))
    if any(mt.t1[1] + mt.t2[1] != size + 1 for mt in first):
        return None
    return sorted(out.values(), key=lambda mt: (mt.r, mt.m))


def round_weeks_for(start: int, n_rounds: int, round_type: int) -> list[list[int]]:
    """Sleeper playoff_round_type: 0 one week per round, 1 two-week championship, 2 all rounds two weeks."""
    out, w = [], start
    for r in range(n_rounds):
        length = 2 if round_type == 2 or (round_type == 1 and r == n_rounds - 1) else 1
        out.append(list(range(w, w + length)))
        w += length
    return out


# --- building from the API ---------------------------------------------------------
def pairings_from(matchups) -> list[tuple[int, int]] | None:
    groups: dict[int, list[int]] = {}
    for mu in matchups:
        if mu.matchup_id is not None:
            groups.setdefault(mu.matchup_id, []).append(mu.roster_id)
    if not groups or any(len(v) != 2 for v in groups.values()):
        return None
    return [(v[0], v[1]) for v in groups.values()]


def build_format(client, league_id: str, snap_week: int, flags: list[str] | None = None) -> LeagueFormat:
    flags = list(flags or [])
    league = client.league(league_id)
    st = league.settings
    rosters = client.rosters(league_id, max_age_s=60)
    ids = sorted(r.roster_id for r in rosters)
    k = int(st.get("playoff_teams") or 6)
    start = int(st.get("playoff_week_start") or 22)
    standings = {}
    for r in rosters:
        s = r.settings
        ties = float(s.get("ties", 0) or 0)
        standings[r.roster_id] = Standing(
            wins=float(s.get("wins", 0) or 0) + 0.5 * ties, losses=float(s.get("losses", 0) or 0) + 0.5 * ties,
            pf=float(s.get("fpts", 0) or 0) + float(s.get("fpts_decimal", 0) or 0) / 100,
            pa=float(s.get("fpts_against", 0) or 0) + float(s.get("fpts_against_decimal", 0) or 0) / 100)
    first_unscored = int(st.get("last_scored_leg", 0) or 0) + 1

    order = sorted(ids, key=lambda i: (-standings[i].wins, -standings[i].pf, standings[i].pa, i))
    seed_of = {rid: s + 1 for s, rid in enumerate(order)}
    raw = None
    try:
        raw = client._get("v1", f"/league/{league_id}/winners_bracket", None, 3600)
    except Exception as e:  # noqa: BLE001 - any failure degrades to the standard bracket
        flags.append(f"bracket_unavailable:{type(e).__name__}")
    bracket = bracket_from_api(raw, seed_of, k) if raw else None
    source = "api"
    if bracket is None:
        bracket, source = standard_bracket(k), "standard_fixed"
        flags.append("bracket_assumed_standard_fixed")
    n_rounds = max(mt.r for mt in bracket)
    round_weeks = round_weeks_for(start, n_rounds, int(st.get("playoff_round_type", 0) or 0))

    pairings: dict[int, list[tuple[int, int]]] = {}
    for w in range(first_unscored, start):
        try:
            pr = pairings_from(client.matchups(league_id, w, max_age_s=3600))
        except Exception:  # noqa: BLE001
            pr = None
        if pr is None:
            flags.append(f"pairings_missing:week_{w}")
            continue
        pairings[w] = pr
    td = st.get("trade_deadline")
    return LeagueFormat(
        roster_ids=ids, playoff_teams=k, first_unscored_week=first_unscored, reg_last_week=start - 1,
        round_weeks=round_weeks, bracket=bracket, pairings=pairings, standings=standings,
        trade_deadline_week=int(td) if td else None, review_days=int(st.get("trade_review_days", 1) or 0),
        flags=flags, bracket_source=source)


# --- trade timing ------------------------------------------------------------------
def effective_date(today: date, review_days: int, accepted_in_days: int = 0) -> date:
    """First NBA game date (US Eastern) a trade accepted ``accepted_in_days`` from today can affect.

    It processes at midnight Pacific after the review ends (design section 2, How to Trade), before
    any game of that day tips.
    """
    return today + timedelta(days=accepted_in_days + review_days + 1)


def processing_time(eff: date, tz: ZoneInfo) -> datetime:
    """Midnight Pacific that starts ``eff``, shown in ``tz`` (18:30 Adelaide in November)."""
    return datetime(eff.year, eff.month, eff.day, tzinfo=PT).astimezone(tz)


def week_of(week_dates: dict[int, list[date]], d: date) -> int | None:
    """Sleeper's Monday-Sunday week containing ``d``, given each known week's game dates."""
    best = None
    for w, ds in sorted(week_dates.items()):
        monday = min(ds) - timedelta(days=min(ds).weekday())
        if monday <= d:
            best = w
    return best


def first_new_week(week_dates: dict[int, list[date]], eff: date, current_week: int,
                   rule: str) -> tuple[int, float]:
    """(first week the new rosters play, share of that week they count).

    ``next_week_if_played`` (conservative default): a trade that lands mid-week changes nothing until
    next week, because a player who already played cannot be used by his new team before then and
    the simulator never credits a partial week the rules might not allow. A trade that lands on the
    week's first day counts from that week (both readings agree). ``immediate`` blends the landing
    week by the share of its days still to come.
    """
    w = week_of(week_dates, eff)
    if w is None:  # beyond the known schedule: nothing to credit this season
        return current_week + 1, 1.0
    ds = week_dates.get(w, [eff])
    monday = min(ds) - timedelta(days=min(ds).weekday())
    if eff <= monday:
        return w, 1.0
    if rule == "immediate":
        left = (monday + timedelta(days=7) - eff).days
        return w, left / 7.0
    return w + 1, 1.0
