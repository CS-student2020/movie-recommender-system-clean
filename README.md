# Movie Recommender System

A production-oriented movie recommendation system designed with a clean, layered architecture and a strong focus on maintainability, testability, and real-world engineering practices.

---

## Status

🚧 **Work in Progress**

This project has undergone a **major architectural refactor** toward a Clean Architecture structure:

- Domain / Core (pure business logic, no I/O)
- Service layer (orchestration)
- API layer (delivery)

Older structures may still exist in the commit history.

---

## What’s Implemented So Far

- Pure Domain/Core recommendation logic (I/O-free)
- User-based and Item-based Collaborative Filtering
- Matrix Factorization (SVD-style) model
- Similarity computation (user-user, item-item)
- Ranking and recommendation orchestration
- Production-ready FastAPI delivery layer
- Offline evaluation with reproducible metrics
- Fully unit-tested core and integration-tested contracts

---

## Tech Stack

- Python
- Pandas / NumPy
- PyTorch (matrix factorization)
- FastAPI
- Pytest

---

## Project Goals

- Demonstrate clean architecture principles in a real ML system
- Separate business logic from infrastructure concerns
- Build a recommender system that is **production-ready**, not tutorial-style
- Apply BigTech-level engineering practices (contracts, tests, evaluation)

---

## Roadmap

- [x] Complete Domain/Core refactor
- [x] Finalize Service layer
- [x] API layer (FastAPI)
- [x] Evaluation and metrics
- [ ] Deployment and observability

---

## Architecture Overview (DDD-Inspired)

This project follows a **Domain-Driven Design (DDD) inspired** architecture with a clear separation of concerns between layers.

### High-level Flow

API (`apps/api`) → Service (`src/recommender/service`) → Domain (`src/recommender/domain`)  
Infrastructure and data modules provide inputs to the service layer.

### Layers

**Domain (`src/recommender/domain/`)**
- Contains pure recommendation and ranking logic.
- All computations are performed in memory.
- **No I/O**:
  - No database access
  - No file reads or writes
  - No logging or external side effects
- Fully deterministic and unit-testable.
- Example modules:
  - `predict_cf.py`
  - `ranking.py`
  - `recommend_for_user.py`

**Service (`src/recommender/service/`)**
- Orchestrates domain logic and coordinates dependencies.
- Prepares inputs for the domain layer.
- Maps domain outputs to structured, typed results.
- Keeps the domain isolated from infrastructure concerns.

**API (`apps/api/`)**
- Delivery layer implemented with FastAPI.
- Exposes HTTP endpoints and request/response schemas.
- Contains no business logic.
- Enforces error contracts and request lifecycle observability.

**Infrastructure / Data**
- Handles persistence and external systems (MongoDB, CSV/JSON, loaders, writers).
- Responsible only for data access and side effects.
- Supplies data to the service layer.
- Precomputed similarity matrices are generated via the data preparation pipeline and are intentionally not versioned.

**Scripts (`scripts/`)**
- One-off runnable utilities and demos (e.g. evaluation runners, Mongo smoke tests).
- Kept outside `src/` to maintain an import-safe package structure.

### Key Design Principle

> The domain layer is independent of I/O.  
> All side effects (databases, files, network, logging) live outside the domain layer.

---

## Evaluation (Offline)

This project includes a reproducible offline evaluation protocol for recommendation quality.

### Protocol (Leakage-safe)

- Split: per-user holdout (leave-one-out)
- For each eligible user (>= 2 interactions):
  - test item = last interaction
  - train = remaining interactions
- Candidates: global train universe, excluding items already seen in train per user

### Metrics

- Precision@K
- Recall@K

### Baselines

- Popularity baseline (global most-popular items)
- Random baseline (deterministic via seed)

### Run evaluation

```bash
make eval
make eval_md
