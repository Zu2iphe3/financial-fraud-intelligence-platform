# Repository Working Agreement

This repository is a learning and portfolio project. Optimize for code quality, clarity, maintainability, and understanding rather than implementation speed.

These instructions apply to the entire repository.

## Repository workflow

1. Begin each unit of work from an up-to-date `main` branch.
2. Create one focused branch for the change.
3. Explain important design decisions before implementing them.
4. Keep changes focused, readable, documented, and tested in proportion to their risk.
5. Inspect all changed files before proposing a commit.
6. Ask for the user's explicit approval before every commit.
7. Ask for the user's explicit approval before every push.
8. Open a pull request only after explaining the change and how it should be reviewed.
9. Merge only after the pull request has been reviewed and approved.

Do not commit directly to `main` after the initial repository setup. Do not combine unrelated work in one branch or pull request.

## Branch naming conventions

Do not use the `codex/` prefix.

Use one of these formats:

- `feature/<feature-name>` for new functionality
- `fix/<bug-name>` for bug fixes
- `docs/<topic>` for documentation-only work
- `refactor/<topic>` for internal restructuring without intended behaviour changes
- `test/<topic>` for test-only work

Use lowercase kebab-case after the prefix. Keep names short, specific, and descriptive.

For tracked backlog work, include the lowercase ticket ID in the branch name:

```text
<type>/<ticket-id>-<short-purpose>
```

Example:

```text
feature/ph3-01-postgres-transaction-storage
```

Examples:

- `feature/paysim-exploration`
- `feature/postgres-ingestion`
- `feature/kafka-producer`
- `fix/duplicate-transaction-ingestion`
- `docs/local-setup`

## Commit message conventions

Use Conventional Commits only:

```text
<type>: <short imperative description>
```

Common types:

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation-only change
- `test:` test addition or correction
- `refactor:` restructuring without an intended behaviour change
- `chore:` tooling, configuration, or maintenance
- `ci:` continuous-integration change

Examples:

- `feat: add PaySim data loader`
- `docs: document PaySim dataset`
- `test: validate transaction schema`
- `refactor: simplify ingestion pipeline`
- `chore: configure development environment`

Keep each commit focused and independently understandable. Do not include generated data, credentials, real `.env` files, raw datasets, or unrelated edits.

## Required approval before every commit

Never create or amend a commit without the user's explicit approval.

Before requesting approval, show the user:

1. The output of `git status`.
2. The output of `git diff --stat` (and the staged equivalent when files are already staged).
3. The proposed Conventional Commit message.
4. The output of `git remote -v`.
5. A concise summary of validation or tests already run.

Wait for an explicit approval response before running `git commit` or `git commit --amend`.

## Required approval before every push

Never push automatically. A previous approval to commit does not authorize a push.

Before requesting push approval, show the user:

- The branch being pushed.
- The commits that are ahead of the remote branch.
- The destination remote and branch.
- The latest test or validation result.

Wait for a separate explicit approval response before running `git push`, including force pushes.

## Pull request process

Before creating a pull request, explain:

1. What files changed.
2. Why each category of file changed.
3. What problem the change solves.
4. What tests and validation were run.
5. How the user should review the change.
6. Any risks, limitations, deferred work, or important design decisions.

Wait for the user's approval before creating the pull request. Use a clear title and a description containing a summary, validation evidence, and review guidance. Do not merge a pull request without explicit approval.

## Review checklist

Before proposing a commit, push, pull request, or merge, confirm as applicable:

- The change matches the requested scope.
- Functions are small, readable, and well documented.
- Names communicate intent and avoid unnecessary abbreviations.
- Important design decisions are explained.
- Tests cover important behaviour and edge cases.
- Relevant tests pass locally.
- Documentation reflects current behaviour.
- Raw data, secrets, credentials, local environments, and generated service state are not tracked.
- `git diff` contains no unrelated or accidental changes.
- The target remote belongs to the user's personal GitHub repository.
- The branch follows the repository naming convention.
- The commit message follows Conventional Commits.
- The pull request is focused and gives clear review instructions.

## Code quality

- Keep functions small and focused on one responsibility.
- Prefer clear, explicit code over clever or compressed implementations.
- Add useful docstrings or comments that explain why important or non-obvious logic exists, especially chunked processing, validation rules, leakage protections, and PaySim-specific calculations.
- Do not comment every obvious line. Prefer concise explanations of purpose, assumptions, constraints, and tradeoffs that help a learner understand the design.
- Explain important design decisions before implementing them.
- Preserve reproducibility for data processing, experiments, and generated reports.
- Favour maintainable project structure and incremental learning over premature optimization.
