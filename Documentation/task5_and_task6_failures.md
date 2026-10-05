# Task 5 and task 6 failures

## Latest run: task 6

Run: `9c4593fbc21943b3946a5f870aa83276`.
Task: five coins, red key, blue key, three more coins, then GOAL.

The final CRM accepts this event sequence. The saved grid has a valid 44-step
ordered route. Each of the three evaluations nevertheless produced an empty
event trace, reward -2, approximately zero reward efficiency, and truncation
at 200 steps. This failure occurred during agent evaluation.

Replaying the saved grid and CRM with the original 5,000 episodes and seed 42
reproduced the last-100-episode mean return of 4.1004 and the failed evaluation.
The greedy policy alternates between `(0, 0)` and `(0, 1)` for all 200 steps.
At the start, RIGHT has Q=9.7050 and DOWN has Q=9.7016; at `(0, 1)`, LEFT
has Q=9.7125. These estimates make the two-cell cycle the greedy choice.
The actual return is -2, far below those learned values.

All three iterations used the same grid, equivalent CRM, training settings,
and seed. Training starts with a fresh Q table each iteration, so the retries
reproduced the same learned policy. Judge calls consumed 351.18 of the total
464.03 seconds without changing the CRM used for training. The first judge
also misinterpreted additive `(0,0)` updates as counter resets and Z as nonzero;
those claims are contradicted by the executable CRM.

The observation contains agent coordinates, machine state, and counters, but
does not include which collectible positions have already been consumed.
Two histories can therefore share the same observation and action while
producing different rewards. This was directly demonstrated at `(1,0)` with
counter values `(4,3)`: moving DOWN yields -0.01 if the coin at `(2,0)` was
already collected, and +1 if a different coin was collected instead.
This hidden collection state is a structural contributor to unstable Q values;
the replay alone does not isolate its effect from insufficient convergence.

The missing reason was a logging omission: only early generation failures wrote
`failure_reason`; completed unsuccessful evaluations did not.

## Task 5

Run: `969c7d47cb7742729d5d2b0ab4b50d58`.
Task adds 11 bell rings and a yellow key after the second coin group.

The first generation placed the yellow key before the bell group. Corrections
fixed that order but introduced an off-by-one counting error. The transition
from State 2 to State 3 consumes the first BELL with counter delta `(0,0,0)`.
Only the remaining ten rings decrement the counter, leaving it at 1 rather
than zero. YELLOW_KEY then fails its zero guard, and GOAL cannot terminate.
All three corrections repeated this structure. No training occurred.

Changing just the entry transition's counter update from `(0,0,0)` to
`(0,0,-1)` makes the saved CRM accept the exact requested event sequence.
This one-line repair is covered by a regression test; it does not establish
that agent training would complete the task.

## Logging changes and verification

Completed failures now record step limits, trace discrepancies, and insufficient
reward efficiency. Abnormal exits record their error and execution stage,
including provider failures, call limits, training errors, and interruptions.
The logger also provides a nonempty fallback if callers omit diagnostics.

Task 6's missing reason was filled from its saved artifacts. Task 5's existing
reason was preserved. A database snapshot was saved before the repair, and
the verification query found zero active unsuccessful runs missing reasons.
All 52 tests passed.

Reproduction evidence is in `outputs/task6_training_replay.json`,
`outputs/task6_observation_aliasing.json`, and `outputs/failure_reason_backfill.json`.
