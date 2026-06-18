# SkillSimulate data + experiment Makefile.
#
# Common targets:
#   make data           — fetch all real datasets + synthetic dev fixtures
#   make data-real      — fetch real datasets only (wikipedia + reddit + github)
#   make data-synthetic — generate small synthetic dev fixtures
#   make data-check     — report which datasets are present
#   make data-clean     — delete data/raw/
#   make smoke          — CADP_MOCK_LLM=1 dev smoke test
#
# Override knobs (set as env vars):
#   WIKI_MAX_THREADS, REDDIT_MAX_SUBMISSIONS, GH_MAX_ISSUES, GH_MAX_EVENTS,
#   SYNTHETIC_N_THREADS, SYNTHETIC_N_USERS, PYTHON
#
# GitHub downloader needs CADP_GITHUB_TOKEN (or GITHUB_TOKEN / GH_TOKEN).
# Wikipedia + Reddit downloaders use ConvoKit corpora (no token required).

PYTHON        ?= python
DATA_DIR      ?= data/raw
GH_TOKEN_ENV  := $(or $(CADP_GITHUB_TOKEN),$(GITHUB_TOKEN),$(GH_TOKEN))

# Default cap values — pass through to downloaders only when set.
WIKI_ARGS     := $(if $(WIKI_MAX_THREADS),--max-threads $(WIKI_MAX_THREADS))
REDDIT_ARGS   := $(if $(REDDIT_MAX_SUBMISSIONS),--max-submissions $(REDDIT_MAX_SUBMISSIONS))
GH_ARGS       := $(if $(GH_MAX_ISSUES),--max-issues-per-repo $(GH_MAX_ISSUES)) $(if $(GH_MAX_EVENTS),--max-events-per-issue $(GH_MAX_EVENTS))
SYNTH_ARGS    := --n-threads $(if $(SYNTHETIC_N_THREADS),$(SYNTHETIC_N_THREADS),50) --n-users $(if $(SYNTHETIC_N_USERS),$(SYNTHETIC_N_USERS),40)

.PHONY: data data-real data-synthetic data-wikipedia data-reddit data-github \
        data-check data-clean smoke help

help:
	@echo "SkillSimulate Makefile targets:"
	@echo "  data           fetch all real datasets + synthetic fixtures"
	@echo "  data-real      real datasets only"
	@echo "  data-synthetic synthetic dev fixtures only"
	@echo "  data-wikipedia just wikipedia (ConvoKit wiki-corpus)"
	@echo "  data-reddit    just reddit (ConvoKit winning-args-corpus)"
	@echo "  data-github    just github (pandas-dev/pandas etc)"
	@echo "  data-check     report present datasets"
	@echo "  data-clean     delete data/raw/"
	@echo "  smoke          mock-LLM dev smoke test (no API cost)"

data: data-real data-synthetic

data-real: data-wikipedia data-reddit data-github

data-wikipedia:
	@if [ -f "$(DATA_DIR)/wikipedia/wiki_wikiconv.jsonl" ]; then \
		echo "[skip] wikipedia already present: $(DATA_DIR)/wikipedia/wiki_wikiconv.jsonl"; \
	else \
		echo "[fetch] wikipedia (ConvoKit wiki-corpus — large, ~200MB)"; \
		mkdir -p "$(DATA_DIR)/wikipedia"; \
		$(PYTHON) scripts/download_wikipedia.py --data-dir "$(DATA_DIR)" $(WIKI_ARGS); \
	fi

data-reddit:
	@if [ -f "$(DATA_DIR)/reddit/reddit_comments.jsonl" ]; then \
		echo "[skip] reddit already present: $(DATA_DIR)/reddit/reddit_comments.jsonl"; \
	else \
		echo "[fetch] reddit (ConvoKit winning-args-corpus, r/changemyview)"; \
		mkdir -p "$(DATA_DIR)/reddit"; \
		$(PYTHON) scripts/download_reddit.py --data-dir "$(DATA_DIR)" $(REDDIT_ARGS); \
	fi

data-github:
	@if [ -f "$(DATA_DIR)/github/github_pandas-dev-pandas.jsonl" ]; then \
		echo "[skip] github already present: $(DATA_DIR)/github/github_pandas-dev-pandas.jsonl"; \
	else \
		if [ -z "$(GH_TOKEN_ENV)" ]; then \
			echo "[warn] no CADP_GITHUB_TOKEN / GITHUB_TOKEN / GH_TOKEN in env — rate-limited (60 req/hr unauth)."; \
			echo "       set a token to get 5000 req/hr."; \
		fi; \
		echo "[fetch] github (pandas-dev/pandas, numpy/numpy, scikit-learn/scikit-learn, python/cpython)"; \
		mkdir -p "$(DATA_DIR)/github"; \
		$(PYTHON) scripts/download_github.py --out-dir "$(DATA_DIR)/github" $(GH_ARGS); \
	fi

data-synthetic:
	@echo "[gen] synthetic dev fixtures (50 threads / 40 users per platform)"
	$(PYTHON) scripts/generate_synthetic_corpus.py $(SYNTH_ARGS) --force

data-check:
	@echo "=== data/raw status ==="
	@for p in wikipedia reddit github; do \
		echo "--- $$p ---"; \
		ls -lh "$(DATA_DIR)/$$p/" 2>/dev/null | grep -vE "^total|^$$" || echo "(missing)"; \
	done

data-clean:
	@echo "[rm] $(DATA_DIR)/"
	@rm -rf "$(DATA_DIR)"
	@mkdir -p "$(DATA_DIR)"

smoke:
	@echo "[smoke] CADP_MOCK_LLM=1 exp1 on configs/dev.yaml (~2 min)"
	CADP_MOCK_LLM=1 $(PYTHON) -m src.main run --config configs/dev.yaml --type exp1
