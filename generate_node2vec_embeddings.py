

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_node2vec_embeddings.py – Generate Node2Vec embeddings only
==================================================================

This script generates ONLY Node2Vec graph embeddings for the SNOMED CT graph.
This is separated from the main pipeline to avoid dependency conflicts.

Requirements:
- node2vec
- networkx
- joblib
- numpy
- tqdm

Author: L. Megret – Inria SED (2025)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import networkx as nx
from tqdm.auto import tqdm

# Graph processing
from node2vec import Node2Vec

def setup_paths():
    """Setup all file paths for the project."""
    path_directory = os.getcwd()
    
    # Input paths
    SNOMED_GRAPH_RELATIVE_PATH = os.path.join("snomed_graph", "full_concept_graph_snomed_ct_int_rf2_20241201.gml")
    
    # Output paths
    DATA_DIR = Path(path_directory) / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    paths = {
        'SNOMED_GRAPH_PATH': os.path.join(path_directory, SNOMED_GRAPH_RELATIVE_PATH),
        'EMBEDDINGS_DIR': EMBEDDINGS_DIR
    }
    
    return paths

def generate_node2vec_embeddings(G, embeddings_dir: Path):
    """Generate Node2Vec graph embeddings with memory-efficient approach."""
    print("Generating Node2Vec embeddings...")
    
    print(f"  Graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    
    # Vérifier que le graphe n'est pas vide
    if len(G.nodes) == 0:
        raise ValueError("Le graphe est vide")
    
    print("  Initializing Node2Vec model...")
    # Utiliser les paramètres de votre notebook
    node2vec_model = Node2Vec(
        G,
        dimensions=512,        # Comme dans votre notebook
        walk_length=100,       # Comme dans votre notebook  
        num_walks=20,          # Comme dans votre notebook
        p=0.5,                 # Comme dans votre notebook
        q=2.0,                 # Comme dans votre notebook (high q)
        workers=min(4, os.cpu_count() or 4)  # Limiter à 4 workers max
    )
    
    print("  Training Word2Vec model on random walks...")
    # Paramètres d'entraînement de votre notebook
    hs = 0
    negative = 10
    epochs = 10
    
    word2vec_model = node2vec_model.fit(
        window=15,             # Comme dans votre notebook
        min_count=1,           # Comme dans votre notebook
        batch_words=256,       # Comme dans votre notebook
        hs=hs,                 # Comme dans votre notebook
        sg=1,                  # Skip-gram (comme dans votre notebook)
        negative=negative,     # Comme dans votre notebook
        epochs=epochs          # Comme dans votre notebook
    )

    print("  Building embedding dictionary...")
    # Construire le dictionnaire directement sans passer par JSON
    embedding_dict = {}
    
    # Utiliser tqdm pour suivre la progression
    for node_label in tqdm(G.nodes(), desc="Extracting embeddings", total=len(G.nodes)):
        # Convertir directement en numpy float32 (plus efficace que tolist() puis conversion)
        embedding_vector = word2vec_model.wv[node_label].astype("float32")
        embedding_dict[str(node_label)] = embedding_vector

    print("  Saving embeddings...")
    pkl_graph = embeddings_dir / "graph_node2vec_high_q.pkl.gz"
    joblib.dump(embedding_dict, pkl_graph, compress=("gzip", 3))
    print(f"    ✅ {pkl_graph}")
    
    return len(embedding_dict)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate Node2Vec embeddings for SNOMED CT")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if file exists")
    args = parser.parse_args()
    
    print("=" * 80)
    print("SNOMED CT Node2Vec Embeddings Generation")
    print("=" * 80)
    
    # Setup paths
    paths = setup_paths()
    
    # Check if output file already exists
    output_file = paths['EMBEDDINGS_DIR'] / "graph_node2vec_high_q.pkl.gz"
    if output_file.exists() and not args.force:
        print(f"✅ Node2Vec embeddings already exist: {output_file}")
        print("Use --force to regenerate")
        return 0
    
    # Check if SNOMED graph exists
    if not os.path.exists(paths['SNOMED_GRAPH_PATH']):
        print(f"❌ Error: SNOMED graph not found: {paths['SNOMED_GRAPH_PATH']}")
        return 1
    
    # Load SNOMED CT graph
    print("Loading SNOMED CT graph...")
    try:
        G = nx.read_gml(paths['SNOMED_GRAPH_PATH'], label='label')
        print(f"Loaded graph with {len(G):,} nodes and {len(G.edges):,} edges")
    except Exception as e:
        print(f"❌ Error loading graph: {str(e)}")
        return 1
    
    # Generate Node2Vec embeddings
    try:
        num_embeddings = generate_node2vec_embeddings(G, paths['EMBEDDINGS_DIR'])
        print(f"\n🎉 Successfully generated {num_embeddings:,} Node2Vec embeddings")
        return 0
        
    except Exception as e:
        print(f"❌ Error generating Node2Vec embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
