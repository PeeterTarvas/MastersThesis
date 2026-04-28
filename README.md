Main fairness applying clustering methods are in:

src/fair_clustering/algorithms/main_bera_fair_clustering.py -> https://arxiv.org/abs/1901.02393

src/fair_clustering/algorithms/main_bercea_fair_clustering.py -> https://arxiv.org/abs/1811.10319

src/fair_clustering/algorithms/main_boehm_fair_clustering.py -> https://pure.au.dk/portal/en/publications/algorithms-for-fair-k-clustering-with-multiple-protected-attribut/

src/fair_clustering/algorithms/main_backurs_fair_clustering.py -> https://arxiv.org/abs/1902.03519

src/fair_clustering/csv_loader.py -> dataset loading and preprocessing

evaluate.py -> common evaluation methods
actual evaluations are inside: src/fair_clustering/evaluations

src/fair_clustering/kmedian.py -> baseline kmedian

building a package: python -m build --wheel
installing package locally: pip install -e .  