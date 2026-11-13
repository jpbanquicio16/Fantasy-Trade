"""Each side's best single follow-up drop-and-add, and the rule that two managers cannot claim the same
free agent (design section 7, "Evaluating complete rosters"). Shared by the whole-league search and the
named-player mode.
"""

from __future__ import annotations

from trade.evaluate import OBJ_KEY, Choice, Evaluator
from trade.search import Screen, Settled


def follow_up(screen: Screen, s: Settled, keep: set[str], k_drop: int = 3, exclude: set[str] = frozenset()) -> tuple[float, Settled]:
    """A side's best single drop-and-add after the trade, if it beats holding (screened value).

    ``keep`` players are never dropped (the incoming ones; a manager does not cut who he just got). Without
    this step the first prototype run reported an owner losing 39.4 FP a week where the real loss was 8.1.
    """
    base = screen.value(s.pids)
    active = [p for p in s.pids if p not in s.il and p not in keep and p not in s.adds]
    best_v, best = base, None
    for d in screen.lowest(active, k_drop):
        rest = [p for p in s.pids if p != d]
        for f in screen.fa_screen:
            if f in s.pids or f in exclude:
                continue
            v = screen.value(rest + [f])
            if v > best_v + 1e-9:
                best_v, best = v, (d, f)
    if best is None:
        return base, s
    d, f = best
    return best_v, Settled([p for p in s.pids if p != d] + [f], s.adds + [f], s.drops + [d], list(s.il))


def complete(screen: Screen, rid: int, out, inn, exclude: set[str] = frozenset()) -> tuple[float, Settled] | None:
    """Settle one side and apply its follow-up move. ``exclude``: free agents the other side already claims
    (two managers cannot both add the same player)."""
    s = screen.settle(rid, out, inn)
    if s is None:
        return None
    return follow_up(screen, s[1], set(inn), exclude=exclude)


def try_follow_up(ev: Evaluator, screen: Screen, ch: Choice, v: float, objective: str, keep: set[str]):
    """Keep your own best follow-up drop-and-add only if simulation says it helps."""
    key = OBJ_KEY[objective]
    _, s2 = follow_up(screen, ch.my, keep)
    best = ev.run(ch, objective)
    if s2 is not ch.my:
        alt = Choice(ch.offer, s2, ch.their, ch.alt)
        r2 = ev.run(alt, objective)
        if float(r2.me[key].mean()) > float(best.me[key].mean()):
            return alt, r2
    return ch, best


def resolve_claims(ev: Evaluator, screen: Screen, ch: Choice, r, objective: str):
    """If the partner's follow-up claims a free agent you also add, he re-picks without him (you chose first)."""
    mine = set(ch.my.adds)
    if not mine & set(ch.their.adds):
        return ch, r
    o = ch.offer
    theirs = complete(screen, o.partner, o.get, o.give, exclude=mine)
    if theirs is None:
        return ch, r
    fixed = Choice(o, ch.my, theirs[1], ch.alt)
    return fixed, ev.run(fixed, objective)
