# --------------------------------------------------------------
# Recommender System — Pipeline Execution Makefile
# --------------------------------------------------------------
# Usage:
#   make similarity_user    → Compute user-user similarity
#   make similarity_item    → Compute item-item similarity
# --------------------------------------------------------------

# Python interpreter
PYTHON=python

# Activate virtual environment (optional, if needed)
VENV=. venv/bin/activate &&

# --------------------------------------------------------------
# User-User Similarity Pipeline
# --------------------------------------------------------------
similarity_user:
	$(VENV) \
	$(PYTHON) -m src.recommender.compute_similarity_user

# --------------------------------------------------------------
# Item-Item Similarity Pipeline
# --------------------------------------------------------------
similarity_item:
	$(VENV) \
	$(PYTHON) -m src.recommender.compute_similarity_item

# --------------------------------------------------------------
# Run both pipelines
# --------------------------------------------------------------
similarity_all:
	$(VENV) \
	$(PYTHON) -m src.recommender.compute_similarity_user
	$(VENV) \
	$(PYTHON) -m src.recommender.compute_similarity_item

# --------------------------------------------------------------
# Clean generated outputs
# --------------------------------------------------------------
clean:
	rm -f data/similarity_matrix_user_user.csv
	rm -f data/similarity_matrix_item_item.csv

