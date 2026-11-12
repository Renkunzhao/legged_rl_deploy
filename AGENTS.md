# AGENTS.md

## Scope

This repository is **hardware deployment code**. It runs on the physical Go2 / G1 and drives real motors.
A mistake here can damage the robot or injure people, so behaviour-changing changes get human review before they
are committed — see the git rules below for exactly when an agent may run a git command.

The workspace-level [`../AGENTS.md`](../AGENTS.md) also applies (manual-visualization policy, deployment
baseline, results layout). This file adds the git rules and takes precedence where the two conflict.

## Git: opt-in only

**Default: no git operations.** Do not touch history, the index, or the remote on your own initiative.

1. **Ask before acting.** `git add`, `commit`, `push`, `rebase`, `reset`, `restore`, `checkout -- <path>`, `stash`,
   `cherry-pick`, `revert`, `merge`, `tag`, and anything that deletes or rewrites files or history: do not run them
   unless the human explicitly asked for that operation in the current turn.
2. **Explicit request is enough — then do it.** If the human says "commit this", "push it", "revert that commit",
   do exactly that operation. No extra warning ritual, no refusing on principle. Just scope it to what was asked:
   the named files, the named branch, the named commit. Never bundle unrelated changes into an explicitly requested
   commit.
3. **Review first for code that changes behaviour.** When the commit is about to include agent-written deploy code
   (controller, safety, config, policy wiring), confirm the human has seen the diff. A request like "commit the
   docs" or "commit what I reviewed" already counts as that confirmation. When in doubt, show the diffstat and ask;
   do not guess.
4. **Report history changes accurately.** After any explicitly requested history operation (reset, revert, rebase,
   amend), state the before/after `HEAD` and what happened to uncommitted work, so the human is never surprised.

Read-only git commands (`status`, `diff`, `log`, `show`, `blame`) are always fine and encouraged — report state
accurately instead of assuming it.

## Working style in this repository

- Prefer handing back a precise summary (file, function, line range, why) over quietly committing; let the human
  ask for the git step.
- If a change is speculative, put it in a separate file or behind a flag so it is easy for the human to drop.
- Never delete or overwrite uncommitted work that is already in the tree; ask first if it looks like someone else's
  work in progress. This includes "cleaning up" a soft reset: uncommitted changes may still be staged, as they are
  right after `git reset --soft`.

## Safety rules for deploy-side changes

- Deployment physics is fixed and calibrated: keep the 2 ms control timestep and the calibrated simulator model.
  Never tune mass, inertia, solver parameters, gravity, friction, timestep, or torque limits to make a demo or an
  evaluation pass. Fix the policy or the test, not the physics.
- Do not weaken safety logic (state machine guards, torque limits, e-stop paths, watchdog timeouts) without an
  explicit request, and call out the change loudly when asked to.
- Sim2sim evaluation is a diagnostic tool, not proof of hardware safety. Say so when reporting results.
- Do not create shell wrappers for interactive visualization / manual gamepad demos; those stay manual.
  Automated evaluation scripts are fine to add and modify.
