# Implementation map

Status against *Sleeper Lock-In Trade Recommender - Technical Design* (11 Nov 2026).

## Built (T0-T3 and the logging half of T5)

| Design | Where | Notes |
| --- | --- | --- |
| §1-2 league facts read, not inferred | `format.py`, `trade settings` | Verified for this league on 11 Nov: 12 teams, 9 starters + 5 bench + 3 IL, trade deadline week 18, 2-day review, 6 playoff teams from week 22, one week per round, fixed bracket with byes for seeds 1-2 (read from `winners_bracket`, standard 1 v 4/5, 2 v 3/6 shape), record/PF/PA seeding. Pick trading is on in the league; picks are excluded, as designed. |
| §2 trade timing | `format.effective_date`, `first_new_week`, `processing_time` | Midnight Pacific after the review, shown in Adelaide time (tested across both clock changes). Rule switch `traded_player_week_rule`. |
| §3 four separate quantities, G(T) | `evaluate.py`, `market.py` | Roster value decides; market value only feeds the tier. |
| §3 objectives | `evaluate.OBJ_KEY` | points / top6 / title selectable; all three reported on every card; "objectives disagree" flagged. |
| §5 shortlist-tier policy | `valuation.py` | Seat by stopping value x availability (max-weight assignment), play each slot with its own thresholds, bench cover for whole-week absences. Non-anticipative: thresholds depend only on a player's own distribution. |
| §5 finalist tier | `recommend.attach_v2` (`--policy v2`) | Re-plays the two traded teams with `waiver.sim.policy.simulate_week` (the lineup design's lock-aware policy). Off by default (slow). |
| §5 current week | `context.World._current_week` | The week in progress is played from the real lock state by the lock-aware policy, identically before and after. |
| §3, §7 settle | `search.Screen.settle*`, `roster_after` | Incoming IL-eligible players take free IL slots; a long roster drops (3 lowest-value candidates, chosen by simulation for you, by screen for the partner); a short roster adds the best free agent or leaves the spot empty. |
| §7 enumeration, screen, prune | `search.screen_offers` | All four shapes, hard filters, deterministic lineup value, beat-your-best-waiver-move, partner-loss tolerance with market veto, package dominance, 10 per partner, ~40 shortlisted. |
| §7 fresh-scenario finalists | `recommend.recommend` | Choices fixed on stream 1, reported on an independent stream 2. |
| §7 tiers, bar, no-trade | `market.tier`, `recommend.passes_bar`, `cards.render` | Bar: gain, SE under a third, 10th percentile above zero. Near-miss shown when nothing qualifies. |
| §10 sensitivity | `recommend.fill_finalist` | Other reading of the trade-week rule, weak management for the other ten teams, partner-held-fixed (rival effect), reversal point for the incoming players' output, free agent lost. |
| §9 logging | `store.py` | recommendations, proposals, responses; observed acceptance per tier. |
| §7 trades for a player you name | `targeted.py`, `trade target` | Acquire / move / free-agent modes; accent- and suffix-insensitive name matching with an ambiguity prompt; offers up to 3-for-2 (acquire) and 2-for-2 (move); each side's best single follow-up drop-and-add (two sides never claim the same free agent); screened trade-off front plus best 8 per shape, up to 80 simulated; price-list candidates re-run on fresh scenarios; protect / max-send / shapes / teams options; CSV of every simulated offer. |
| §10 correctness tests | `tests/test_core.py` | Stopping premiums match the lineup design (0.40, 0.63), null trade is exactly zero, unrelated teams' weekly scores unchanged, trade-week rule, bench cover, settle legality, bracket and seeding, timing, tiers. |

| §7 when a manager asks for one of your players | `interest.py`, `trade inbound` | Judges his entered offer with the same paired simulation; three counter kinds (ask for one more, swap the piece, add one of yours for an upgrade), screened then simulated; counters ranked by smallest market concession; full search of every deal with him (1-3 of his players, up to `--max-extra` more of yours); "keep him" when nothing clears the bar; stale offers rejected; request logged in `proposals` with him as sender. Deviation: his offer is entered as player names on the command line (Sleeper exposes no proposals). |
| §7 trade-off chart | `chart.py` | Dependency-free SVG written beside the CSV by `trade target`: your gain vs the owner's, the bar, the front highlighted. |
| §7 follow-up moves and one claim per free agent, whole-league | `followup.py`, `search.screen_offers`, `recommend.recommend` | Both sides' best drop-and-add is in the screen (partner) and the simulation (you); two sides never claim the same free agent. Shared with `target`. |
| §8 cards and nightly run | `app/streamlit_app.py` (`trade app`), `trade nightly`, `scripts/nightly.sh`, `scripts/com.trade.nightly.plist` | Cards read from the SQLite log; "I proposed this" logs a proposal. The app is untested beyond the `store.stored_cards` reader. |

## Deviations (and why)

* **Bracket**: the design assumed a fixed bracket with byes; the API confirms it, so it is read, not assumed.
* **`immediate` trade-week rule** blends the landing week by the share of days left rather than modelling partial
  weeks; the default (`next_week_if_played`) never credits a partial week.
* **Market value** is Sleeper's `search_rank` mapped to FP/G by a log-rank fit (an isotonic fit was tried and
  rejected: piecewise constant, so neighbouring ranks tied). Draft pick/round come from this league's draft.
* **Ability uncertainty** is the waiver core's per-batch rate draw (model band), not a separate per-scenario draw.
* **Projections** are the waiver core's (shrunken EW rates, availability table, role scenarios) held constant out
  to the final; the far horizon does not revert to a prior or widen.

## Not built

* **T4 replay on real seasons** (synthetic leagues, baselines of §10, untouched 2025-26 season). Needs history
  before 2024 for the validation design; the shipped data cover 2024-2026. The scoring-engine and projection
  evaluations live in `waiver evaluate`.
* **Weekly shadow replay** of unmade recommendations (the logging exists; the counterfactual scorer does not).
* **Injury-hazard halved/doubled** sensitivity, **market-informed vs market-independent projection** ablation
  (§6), model-uncertainty draws beyond the sampler's batches (30 in the design).
* **DuckDB/Parquet and `known_at` bitemporal tables** (the waiver
  core's snapshots stand in; the perturbation test is not ported).
* **LightGBM, PyMC, hierarchical acceptance model, beam search, news extraction**: deferred or excluded by the design.
* **Three-team and 3-for-3 trades**, **draft-pick trading**: out of scope by the design.

## Open rules (settle by observation)

1. Can a player who already played this week be used by his new team before next week? (switch above)
2. Can the old team still lock an outgoing player's open score after processing? (card says to lock first)
3. 40+/50+ stacking and technical-foul definition are settled in `waiver` (stack, Sleeper `tf`/`ff`).
