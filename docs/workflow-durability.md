# Making workflow fan-out survive a budget cut

Written 23 July 2026, after a three-track workflow lost 3 of 4 agents to a
session limit and left nothing on disk.

## What actually happened

Run `wf_65b4f4ca-b81` fanned out three tracks (roofs, uvmap, texturing), each
`design -> implement -> 3 parallel verifiers`. Fifteen agents, all at
`effort: high`. Three died on the session limit; one design survived.

Measured from the run's transcripts:

| Agent | Transcript | Tool calls | Text produced | Outcome |
|---|---|---|---|---|
| design:roofs | 267 KB | 8 reads | 143 chars | died mid-read |
| design:texturing | 291 KB | 15 | 15,722 chars | survived |
| build:texturing | 50 KB | 0 | 58 chars | died before starting |
| design:uvmap | 270 KB | 12 reads | 95 chars | died mid-read |

**The failure is not that agents died. It is what they were doing when they
died.** Each one independently re-read the same core modules — `massing.py`,
`osm.py`, `materials.py`, `frame.py`, `test_massing.py`,
`atlas_max_handlers.py` — burning roughly 270 KB apiece before synthesising
anything. Four agents, ~800 KB of duplicated reading, and two of them died with
nothing to show for it.

The surviving agent proves the design was sound: given the same brief it
produced a genuinely useful 15 K design that caught a real gap in my own
briefing. The problem is the cost structure around it, not the fan-out itself.

## Five changes, in order of value

### 1. Pre-digest the context once (biggest single saving)

The orchestrator reads the repo **once** and passes a distilled brief. Agents
should receive facts, not a reading list. Only an agent that genuinely needs to
see a file — a verifier checking real code — should open one.

Concretely: extract the public API surface, the conventions, and the measured
host facts into a `CONTEXT` string and a committed
`docs/agent-brief-<track>.md`. This removes 3/4 of the reading cost outright.

The previous run already half-did this — the brief carried measured OSM
coverage and V-Ray capability lists — and the agents still re-read everything,
because the brief also *told* them to ("Read at minimum: ..."). That
instruction is the bug. Replace it with "these facts are established; do not
re-derive them".

### 2. Artifacts before prose

Every agent must write its work product to a **file** as its first substantive
act, then return a short summary. A design agent that returns 15 K of prose has
produced something that dies with the run; a design agent that writes
`docs/design-roofs.md` and returns "written, 4 open questions" has produced
something durable.

This is the difference between the surviving design being recoverable only by
me parsing `journal.jsonl`, and it simply being a file in the repo.

### 3. Budget-aware fan-out

The Workflow runtime exposes `budget.total`, `budget.spent()` and
`budget.remaining()`. The previous run used none of them. Size the fleet from
what is actually left:

```js
const FLEET = budget.total
  ? Math.max(1, Math.floor(budget.remaining() / 120_000))
  : 3
const tracks = TRACKS.slice(0, FLEET)
log(`budget allows ${FLEET} of ${TRACKS.length} tracks; deferring ${TRACKS.slice(FLEET).map(t => t.key)}`)
```

And check between phases, so a run that is going to die does so *after*
finishing one track rather than halfway through three:

```js
if (budget.total && budget.remaining() < 80_000) {
  log('insufficient budget for the verify phase; stopping with builds complete')
  return results
}
```

### 4. Sequence tracks; parallelise only within one

Three tracks in flight means a budget cut lands on all three. One track at a
time means a cut costs the current track and leaves the finished ones intact.
Parallelism inside a track (the three verifier lenses) is cheap and bounded.

Wall-clock is worse. Salvage rate is much better, and salvage rate is what
failed here.

### 5. Resume instead of restart

`Workflow({scriptPath, resumeFromRunId})` replays completed `agent()` calls
from cache. After the limit resets, the surviving texturing design would have
replayed for free and only the dead agents would have re-run. This was
available and unused.

## Effort tiers

`effort: 'high'` on all fifteen agents was indiscriminate. A reasonable split:

| Stage | Effort | Why |
|---|---|---|
| design | high | the reasoning that matters |
| implement | high | correctness-critical |
| verify (math) | high | adversarial, needs depth |
| verify (host names) | low | mechanical cross-check against a list |
| verify (style/fit) | medium | judgement, but bounded |

The host-name verifier in particular is a literal set-membership check. It does
not need a high-effort budget.

## The check that would have caught this

Before launching, ask: **if this run is cut off halfway, what is on disk?**

For run `wf_65b4f4ca-b81` the answer was "nothing", and that was knowable in
advance — no agent was instructed to write a file until the implement stage.
Any workflow whose answer is "nothing" needs change 2 before it runs.
