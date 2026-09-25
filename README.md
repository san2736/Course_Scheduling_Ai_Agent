# Multi-Agent Bargaining for University Course Scheduling

Applied AI (RL & MAS) mini project. Department and room agents learn to
negotiate a conflict-free weekly timetable using **Expected SARSA**,
implemented by hand — no RL library.

## Install and run

```bash
pip install numpy matplotlib pytest
python -m pytest tests -q               # 7 tests, all passing
python run_experiments.py --episodes 800 --load 90
python run_experiments.py --episodes 800 --load 90 --sweep
```

Results land in `results/report.json` plus `.npy` learning curves and a
sample timetable.

## Layout

| file | what it holds |
|---|---|
| `scheduler/scenario.py` | Session / Faculty / Room data model, seeded generator |
| `scheduler/grid.py` | the 120-cell grid, feasibility engine, validator |
| `scheduler/env.py` | negotiation rounds, clash resolution, swaps, metrics |
| `scheduler/agents.py` | DepartmentAgent, RoomAgent, state encoder, Expected SARSA |
| `scheduler/baselines.py` | random, greedy first-fit, greedy-on-preference |
| `scheduler/train.py` | training loop + Q-learning comparison arm |
| `run_experiments.py` | runs everything and writes results |
| `tests/test_core.py` | constraint and termination tests |

## Design notes worth defending in the viva

**Sessions, not courses.** The atomic unit is a Session (one class, one
subject, one faculty, one room, one slot). Labs and batch splits can be
added later as extra Sessions without rewriting anything.

**Hard constraints are masked, not punished.** `Grid.legal_cells()`
returns only slots that satisfy faculty availability, room capacity, the
per-department daily cap, and no-double-booking. An agent physically
cannot propose an illegal slot, so every timetable produced is valid
(`validate()` returns `[]` on every run). The **−20 clash penalty** is for
a different thing: two departments bidding for the same *free* cell in the
same round. That is competition, not rule-breaking, and it is what the
agents must learn to avoid.

**The compact state.** Agents do not see the raw grid. Each learns from
four coarse features (pending count, good options left, best available
preference, rounds left) giving ~162 states. A Q-table over the full grid
would have astronomically many rows and would never be filled.

**Strategy-level actions.** The five actions are negotiation strategies
(best / safe / urgent / concede / wait), not individual cells. The chosen
strategy then picks a concrete cell. This is what keeps the action space
at 5 instead of 120.

**Feasibility ceiling.** 6 departments × 5 days × 3 classes per day = **90
sessions maximum**. Loads above that are infeasible for any method — we
confirmed no algorithm places more than 90 at a load of 105. So 90 is the
stress point, not an arbitrary number.

## Results (load 90 = the feasibility ceiling, 800 episodes)

| method | placed | preference | fairness (worst dept) | rounds |
|---|---|---|---|---|
| Random | 90/90 | 2.48 | 2.34 | — |
| Greedy first-fit | 89/90 | 2.98 | 2.72 | 21 |
| Greedy on preference | 90/90 | **3.63** | **3.44** | 13 |
| **Learned agents** | 90/90 | 3.52 | 3.28 | 19 |

Learning clearly happens: mean reward rises from about −110 to +215, and
success rate from 0.44 to 0.96 across training. Zero constraint violations
in every run.

**Honest reading: the agents match greedy-on-preference on feasibility but
do not beat it on preference.** At 50% load the gap is similar (3.96 vs
4.11). Across seeds the picture is mixed — at seed 1 and 2, greedy-on-
preference strands 3 and 1 sessions respectively while the agents place
nearly all, so the agents are more *reliable*, but their slot quality is
consistently a little lower.

This is a legitimate result to present. The agents learn a robust
negotiation policy from scratch, with no knowledge of the rules, and land
close to a hand-written heuristic that was given the preference function
directly.

## Known issues / next steps

1. **Swaps barely fire** (`swaps` ≈ 0.05 per episode). The acceptance rule
   was loosened and a `swap_bonus` added, which got it off zero, but the
   bargaining channel is still nearly unused. Fixing this properly is the
   single highest-value next step — it is a headline feature on the slides
   and the mechanism that should rescue boxed-in sessions.
2. **Preference weight.** Raising `Rewards.preference` from 1.0 to 3.0
   improved things a lot. It is probably still too low relative to the
   +10 placement reward; try 5.0 and re-run.
3. **Variance across seeds** — success at load 90 ranges from 0.47 to 1.00
   depending on seed. Run 5+ seeds and report mean ± std rather than a
   single number.
4. **Room agent learning is shallow.** Its state is only
   (contention, how full I am). Adding "how urgent is the best bidder"
   would give it something more interesting to learn.
5. **Scale-up** would need function approximation instead of Q-tables.

## Scenario-design lesson worth reporting

An early version had only one room able to hold 110-student classes. Those
sessions got structurally stranded, success capped at 0.51, and it looked
like an algorithm failure. It was a **scenario generation flaw**. Widening
the room mix lifted success to 0.89 with no change to the agents. Worth a
line in the report: always check whether a bad result is the problem or
the problem instance.
