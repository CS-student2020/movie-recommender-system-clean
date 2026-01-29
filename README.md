# Movie Recommender System

A production-oriented movie recommendation system designed with a
clean, layered architecture and a strong focus on maintainability,
testability, and real-world engineering practices.

## Status
🚧 **Work in Progress**

This project is currently undergoing a **major architectural refactor**
toward a Clean Architecture structure:
- Domain / Core (pure business logic, no I/O)
- Service layer (orchestration)
- API layer (delivery)

Older structures may still exist in the commit history.

---

## What’s implemented so far
- Pure Domain/Core recommendation logic (I/O-free)
- User-based & Item-based Collaborative Filtering
- Matrix Factorization (SVD-style) model
- Similarity computation (user-user, item-item)
- Ranking & recommendation orchestration
- Unit-tested core modules

---

## Tech Stack
- Python
- Pandas / NumPy
- PyTorch (for matrix factorization)
- Pytest

---

## Project Goals
- Demonstrate clean architecture principles in a real ML system
- Separate business logic from infrastructure concerns
- Build a recommender system that is **production-ready**, not tutorial-style

---

## Roadmap
- [ ] Complete Domain/Core refactor
- [ ] Finalize Service layer
- [ ] API layer (FastAPI)
- [ ] Evaluation & metrics
- [ ] Deployment & observability

