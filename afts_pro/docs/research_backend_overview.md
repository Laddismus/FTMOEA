# Research Backend Overview (M1 Hardened)

## Scope
- FastAPI-based backend for research workflows:
  - Backtests with FTMO RiskGuard
  - Strategy Builder (graph + python)
  - Experiments & leaderboards (backtest + RL)
  - RL core (runs, policies) and RL experiments
  - Governance Model Hub (promotion stages)

## Setup (Dev)
1. Create/activate virtual env (python >=3.11).
2. Install deps:
   ```bash
   pip install -r requirements/research.txt
   ```
3. Run server:
   ```bash
   uvicorn research_lab.backend.app.main:app --reload
   ```

## API Overview
- `/api/health`
- `/api/analytics/*`
- `/api/strategy-builder/*`
- `/api/python-strategies/*`
- `/api/backtests/*`
- `/api/experiments/*`
- `/api/rl/*`
- `/api/rl-experiments/*`
- `/api/governance/*`
- OpenAPI docs at `/docs` or `/redoc`.

## Error & Logging
- Errors return `{"detail": "...", "error_code": "...", "request_id": "..."}`.
- Request-level correlation via `X-Request-ID`; logged with method/path/status/duration.

## Notes
- Designed for M1; UI and deeper integrations (M2+) can consume these APIs directly.
