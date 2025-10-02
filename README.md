# MASTER-2-INTERNSHIP---SNOMED-CT-Translation-Experimentation

# NRC‑traduction‑snomed‑LLM

> **Few‑shot translation of SNOMED CT preferred terms (English → French / Spanish) with large language models**

---

## Table of contents

1. [Project overview](#overview)
2. [Folder layout](#layout)
3. [Prerequisites](#prereq)
4. [Installation](#install)
5. [Execution pipeline](#pipeline)
     5‑a. [Step 0 – build the graph](#step0)
     5‑b. [Step 1 – Node2Vec embeddings](#step1)
     5‑c. [Step 2 – Translation & lexical embeddings](#step2)
     5‑d. [Step 3 – Example retrieval](#step3)
     5‑e. [Step 4 – Five‑shot generation](#step4)
6. [Generated artefacts](#artefacts)
7. [Troubleshooting & FAQ](#faq)
8. [License & citation](#license)

 ## 1 – Project overview This repository contains a **reproducible research pipeline** that compares several example‑retrieval strategies for translating SNOMED CT concepts with an instruction‑tuned multilingual LLM (Aya‑101).
The experiment:

* extracts **French** and **Spanish** preferred terms (PT) from their respective National Extensions;
* computes four families of similarity metrics (BoW n‑gram, TF‑IDF, Sentence‑Transformer, Node2Vec);
* builds mixed graph/semantic **few‑shot prompts** (5 exemplars) for every strategy;
* asks the model to translate new English PTs and logs the outputs for further evaluation.

 ## 2 – Folder layout

```
├── data/                     # auto‑created; all outputs land here
│   ├── embeddings/           # *.pkl.gz embedding dictionaries
│   └── five_shot_generations # final TSV translations
├── snomed_graph/             # cached GML graph of the International Edition
├── SnomedCT_FR/              # French Extension (validated translations)
├── SnomedCT_SpanishRelease‑es_PRODUCTION_20240930T120000Z/
├── generate_node2vec_embeddings.py
├── generate_node2vec.sh                   # SLURM wrapper
├── load_translations_embedding.py          
├── retrieval_example.py
├── generate_five_shot.py
└── snomed_pipeline_translations.sh         # SLURM wrapper for full pipeline
```

*Only the key research scripts are listed above – notebooks and helper utils omitted for brevity.*

 ## 3 – Prerequisites ### 3.1 Datasets

| What                                                                         | Where to put it                                                                                                      | Notes                               |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| **SNOMED CT International Edition** (RF2, 2024‑12 release used in the paper) | *anywhere* → convert once to `full_concept_graph_snomed_ct_int_rf2_20241201.gml` inside `snomed_graph/` (see Step 0) | required                            |
| **French validated translations** `traductions_validées_20241218_v2.csv`     | `SnomedCT_FR/`                                                                                                       | required                            |
| **Spanish Extension 2024‑09** (Full + Snapshot folders)                      | `SnomedCT_SpanishRelease‑es_PRODUCTION_20240930T120000Z/`                                                            | required                            |
| **default_example.csv** (hand‑crafted seed examples in FR)                   | project root                                                                                                         | optional – improves default prompts |

### 3.2 Software – two Conda environments `node2vec` still relies on **NetworkX < 3.0** while `snomed_graph` needs **NetworkX ≥ 3.2**.  Mixing the two breaks both, therefore we keep them **separate**:

| Environment              | Purpose                        | Main deps                                                                          |
| ------------------------ | ------------------------------ | ---------------------------------------------------------------------------------- |
| **env_node2vec**         | graph walks & Node2Vec fitting | node2vec, networkx 2.x, gensim                                                     |
| **env_snomed_trans_poc** | everything else                | snomed_graph, torch, sentence‑transformers, faiss‑cpu, transformers, Aya‑101, etc. |

Minimal recipes are provided in `envs/` (feel free to tweak CUDA versions).

 ## 4 – Installation (one‑off)

```bash
# clone the repo
$ git clone https://github.com/<you>/NRC-traduction-snomed-LLM.git
$ cd NRC-traduction-snomed-LLM

# create both envs
$ conda env create -f envs/env_node2vec.yml
$ conda env create -f envs/env_snomed_trans_poc.yml
```

On HPC clusters using **SLURM**, the two submission scripts already activate the right environment; on a local workstation just `conda activate` them manually before each step.

 ## 5 – Execution pipeline The whole run boils down to **two commands** – the first builds Node2Vec embeddings, the second drives the remainder of the experiment.

```bash
# STEP 1 – run once (≈ 24 h on 30 k nodes)
$ sbatch generate_node2vec.sh           # OR: conda activate env_node2vec && python generate_node2vec_embeddings.py

# STEP 2 – full translation pipeline (GPU recommended)
$ sbatch snomed_pipeline_translations.sh             # default (complete run)
$ sbatch snomed_pipeline_translations.sh test        # tiny dry‑run
$ sbatch snomed_pipeline_translations.sh skip-embeddings  # reuse existing *.pkl.gz
```

Below is the detailed flow for curious minds.

 ### Step 0 – Build the concept graph *(one‑off)* If you do **not** have `snomed_graph/full_concept_graph_snomed_ct_int_rf2_20241201.gml` yet:

```python
from snomed_graph.snomed_graph import SnomedGraph
SnomedGraph.from_rf2_folder("/path/to/SnomedCT_International_RF2") \
           .to_serialized("snomed_graph/full_concept_graph_snomed_ct_int_rf2_20241201.gml")
```

This step can be executed inside **env_snomed_trans_poc**.

 ### Step 1 – Node2Vec embeddings *(env_node2vec)*

| Script                                                 | Inputs                                  | Outputs                                        |
| ------------------------------------------------------ | --------------------------------------- | ---------------------------------------------- |
| `generate_node2vec_embeddings.py` *(called by **``**)* | `snomed_graph/full_concept_graph_*.gml` | `data/embeddings/graph_node2vec_high_q.pkl.gz` |

Parameters (`walk_length=100`, `num_walks=20`, `dims=512`, *p* = 0.5, *q* = 2) match the notebook.

 ### Step 2 – Load translations & lexical embeddings *(env_snomed_trans_poc)*

| Script                           | Inputs                                         | Outputs                                                                 |
| -------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| `load_translations_embedding.py` | French & Spanish extension files, SNOMED graph | `data/all_translations.csv`                                             |
|                                  |                                                | `data/samples.csv`                                                      |
|                                  |                                                | BoW, TF‑IDF, ST embeddings (`data/embeddings/*.pkl.gz` except Node2Vec) |

Flags:

* `--skip-embeddings` → only refresh CSVs
* `--force` → overwrite everything

 ### Step 3 – Retrieve in‑context examples *(env_snomed_trans_poc)*

| Script                 | Inputs                                             | Outputs                                         |
| ---------------------- | -------------------------------------------------- | ----------------------------------------------- |
| `retrieval_example.py` | graph, CSVs, all embeddings, `default_example.csv` | `data/default_example_fr_es.csv`                |
|                        |                                                    | `data/example_retrieval_es_fr.csv` (≈ 2 M rows) |

The file contains both *graph*‑based and *vector*‑based neighbours with scores and provenance.

 ### Step 4 – Generate five‑shot prompts & translations *(env_snomed_trans_poc, GPU 11 GB +)*

| Script                                                        | Inputs                                                        | Outputs                                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `generate_five_shot.py`                                       | graph, translations CSVs, example retrieval, default examples | `data/five_shot_generations/five_shot_<method>.tsv` for seven methods: |
| `bow`, `tfidf`, `enc`, `graph`, `rgraph`, `default`, `random` |                                                               |                                                                        |

Useful options:

* `--test` – limit to 20 concepts per method (quick sanity check)
* `--batch-size`, `--max-new` – control Aya‑101 generation

 ## 6 – Generated artefacts

```
data/
├── all_translations.csv           # 6.8 M rows × concept features
├── samples.csv                    # balanced evaluation sample (≈ 7 k rows)
├── embeddings/
│   ├── bow_binary_ngram.pkl.gz
│   ├── tfidf_ngram.pkl.gz
│   ├── st_multilingual.pkl.gz
│   └── graph_node2vec_high_q.pkl.gz
├── default_example_fr_es.csv      # seed examples enriched with ES PT
├── example_retrieval_es_fr.csv    # every candidate neighbour + metadata
└── five_shot_generations/
    ├── five_shot_bow.tsv
    ├── five_shot_tfidf.tsv
    ├── five_shot_enc.tsv
    ├── five_shot_graph.tsv
    ├── five_shot_rgraph.tsv
    ├── five_shot_default.tsv
    └── five_shot_random.tsv
```

Each TSV contains: `sctid`, `preferred_term`, `aya_translation`, reference translation, prompt text, etc.

 ## 7 – Troubleshooting & FAQ

* **ImportError: *****No module named ‘networkx.drawing’*** – you probably mixed the two environments.  Double‑check with `conda list | grep networkx`.
* **GPU OOM during Aya‑101 generation** – use `--batch-size 8` or fall back to CPU (automatic) – it will just be slower.
* **Need to regenerate only TF‑IDF embeddings** – `python load_translations_embedding.py --force --skip-embeddings` then delete `tfidf_ngram.pkl.gz` and rerun without `--skip-embeddings`.
* **Where do results of the paper come from?** – see `ANALYSIS.ipynb` for replication of all metrics.

 ## 8 – License & citation Code is released under the **Apache 2.0** license.  SNOMED CT data is © SNOMED International and distributed under the SNOMED CT Affiliate License; you must hold a valid license to use the terminology files.

If you use this pipeline, please cite:

```text
Megret L., 2025. Few‑shot Translation of SNOMED CT with Large Language Models. Inria SED, Technical Report.
```

Happy translating! 🎉
