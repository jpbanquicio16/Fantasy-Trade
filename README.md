# Fantasy-Trade

Trade recommender for Sleeper **Lock-In** fantasy basketball. It simulates the rest of the season for the
whole league twice - as it stands, and after each candidate trade (required drops and free-agent adds
included) - on shared random scenarios, ranks offers by *your* gain, shows the other manager's gain, and
treats acceptance as a separate, openly heuristic tier. "No worthwhile trade" is a normal answer.

Built from *Sleeper Lock-In Trade Recommender - Technical Design* (11 Nov 2026). See
[`docs/implementation.md`](docs/implementation.md) for what is implemented, what deviates, and what is not built.

It reuses the sibling `../waiver-bot` package (`waiver`) as its core: scoring engine, projections, scenario
sampler, lock-aware policy, league snapshot and cached data. Install that first.

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install -e ../waiver-bot -e ".[dev]"
cp configs/trade.example.yaml configs/trade.yaml
(cd ../waiver-bot && make data fit)       # data and projection parameters, if not done yet
.venv/bin/trade settings                  # T0: league facts read from Sleeper, each marked read / open
.venv/bin/trade recommend                 # about 4-6 minutes at the defaults
.venv/bin/trade recommend --objective title --partner 5 --shapes 1-1,2-1
.venv/bin/pytest
```

## Commands

| Command | Does |
| --- | --- |
| `trade settings` | Slots, IL, deadline, review period, playoff weeks, bracket, seeding - read from the API, with open items marked |
| `trade recommend` | Enumerate, screen, prune, simulate the shortlist, re-evaluate finalists on fresh scenarios, print cards |
| `trade propose REC OFFER` | Log that you sent an offer (the only acceptance labels that can exist) |
| `trade respond ID accepted\|rejected\|countered\|expired\|withdrawn` | Log what happened |
| `trade log` | Proposals, responses, observed acceptance per tier |

Options for `recommend`: `--objective points|top6|title`, `--partner ROSTER_ID`, `--shapes 1-1,2-1,1-2,2-2`,
`--scenarios N`, `--finalist-scenarios N`, `--policy shortlist|v2`, `--rule next_week_if_played|immediate`,
`--no-refresh`, `--json`. Defaults live in `configs/trade.yaml`.

## Reading a card

* **Your change**: expected weekly points (regular season), playoff-week points, P(top 6) and P(title) in
  percentage points, each a *paired* difference with its standard error. The band is the 10th-90th percentile of
  scenario batches (each batch shares one draw of every player's rates).
* **Beats**: your best waiver-only move (or hold). The reported gain is G(T) = best after-trade roster minus the
  best you could do without the trade.
* **Tier** (heuristic, uncalibrated; no probability is shown): from the signs of the partner's *projected* change
  and *market-perceived* change.
* **Reverses if**: how far the incoming players' output must fall to erase the gain; the other reading of the
  trade-week rule; the other ten teams managing worse.
* Only offers with gain above the bar (0.5 FP/wk for `points`), standard error under a third of it, and a positive
  10th percentile are shown.

## Layout

```
src/trade/
  format.py      league format from the API: standings, pairings, bracket, rounds, deadline, trade timing
  valuation.py   Lock-In stopping thresholds, realised weekly values, lineups with bench cover (shortlist tier)
  season.py      matchups, standings, seeding, bracket -> P(top 6), P(title)
  context.py     Context (waiver core) and World (one shared scenario stream + every team's baseline)
  search.py      offers, roster settling (IL, drops, adds), deterministic screen, pruning, dominance
  evaluate.py    paired both-side evaluation, G(T), waiver-only baseline
  market.py      market-perceived value and the compatibility tier
  recommend.py   the pipeline, bar, sensitivity runs
  cards.py       text cards and the "no worthwhile trade" report
  store.py       SQLite log: recommendations, proposals, responses
```
