# Contributing Guide

## Workflow
1. Fork & create a feature branch
2. Add/update tests (must pass `uv run pytest -q`)
3. Update relevant docs
4. Open a PR describing motivation, changes, impact

## Code Style
- Keep functions small & cohesive
- Any new public API must update module docs

## Testing Expectations
- Cover success & failure paths
- Subprocess / network tests must run on CI without GPUs

