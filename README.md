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
| `trade target PLAYER` | Offers for one player you name: *acquire* him if another manager owns him, *move* him if you do. Price list by offer shape, your-gain vs owner-gain front, cards |
| `trade inbound MANAGER PLAYER` | A league-mate asked for one of your players: judges his offer (`--offer NAME...`), builds counters close to it, searches every deal with him, or says keep him. Logs his request as a proposal |
| `trade nightly` | Refresh, search, store, write `data/reports/latest.txt` (schedule with `scripts/nightly.sh` / the launchd plist) |
| `trade app` | Streamlit cards from the stored recommendation (`pip install -e ".[app]"`) |
| `trade propose REC OFFER` | Log that you sent an offer (the only acceptance labels that can exist) |
| `trade respond ID accepted\|rejected\|countered\|expired\|withdrawn` | Log what happened |
| `trade log` | Proposals, responses, observed acceptance per tier |

Options for `recommend`: `--objective points|top6|title`, `--partner ROSTER_ID`, `--shapes 1-1,2-1,1-2,2-2`,
`--scenarios N`, `--finalist-scenarios N`, `--policy shortlist|v2`, `--rule next_week_if_played|immediate`,
`--no-refresh`, `--json`. Defaults live in `configs/trade.yaml`.

`trade target` options: `--protect NAME...` (never given up), `--max-send N` (default 3), `--shapes give-get,...`
(acquire default `1-1,2-1,3-1,1-2,2-2,3-2`; move default `1-1,2-1,1-2,2-2`), `--teams IDS` (move mode),
plus the `recommend` options. Names ignore accents, suffixes and punctuation; an ambiguous name lists ids to
rerun with. A free agent needs no trade and prints the drop a claim would pair with. All simulated offers are
written to `data/reports/target_*.csv`, with the trade-off chart as a matching `.svg`; the printed recommendation id works with `trade propose`.

`trade inbound` options: `--offer` (players he offered; omit if none), `--protect NAME...` (never added to a counter),
`--max-extra N` (extra players of yours in a counter or deal, default 1), `--counters-only`, `--no-log`, `--note`,
plus the `recommend` options. `MANAGER` is a roster id or part of his display name. Counters that clear your bar and are
not in the *unlikely* tier are ordered by the smallest concession asked of him in market terms. His request is logged in
`proposals` with his roster as sender; record your answer with `trade respond ID ...`. An offer naming a player no longer on
either roster is rejected as stale.

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
  interest.py    inbound mode: judge his offer, counters near it, full search with him
  targeted.py    named-player mode: find, enumerate, follow-up moves, front, price list
  recommend.py   the pipeline, bar, sensitivity runs
  cards.py       text cards and the "no worthwhile trade" report
  store.py       SQLite log: recommendations, proposals, responses
```
