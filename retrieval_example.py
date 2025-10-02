"""
retrieval_example.py – Example retrieval for SNOMED CT translations
==================================================================

This script performs example retrieval for SNOMED CT concept translation using both
similarity-based methods (vectorial representations) and graph-based methods 
(relationships in the SNOMED CT graph).

The script performs the following steps:
1. Load required data files (graph, translations, samples, embeddings)
2. Create default_example_fr_es.csv with Spanish preferred terms added
3. Generate comprehensive example retrieval combining:
   - Similarity-based examples (BoW, TF-IDF, SentenceTransformer, Node2Vec)
   - Graph-based examples (parents, siblings, attributes, ancestors)
4. Output single file example_retrieval_es_fr.csv with all examples

Author: L. Megret – Inria SED (2025)
"""

from __future__ import annotations

import os
import ast
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Set, List, Any, Tuple, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
from tqdm.auto import tqdm
from collections import deque

# Graph processing
from snomed_graph.snomed_graph import SnomedGraph

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------

def setup_paths():
    """Setup all file paths for the project."""
    path_directory = os.getcwd()
    
    # Input paths
    SNOMED_GRAPH_RELATIVE_PATH = os.path.join("snomed_graph", "full_concept_graph_snomed_ct_int_rf2_20241201.gml")
    ALL_TRANSLATIONS_RELATIVE_PATH = os.path.join("data", "all_translations.csv")
    SAMPLES_RELATIVE_PATH = os.path.join("data", "samples.csv")
    EMBEDDINGS_DIR_RELATIVE_PATH = os.path.join("data", "embeddings")
    DEFAULT_EXAMPLE_RELATIVE_PATH = os.path.join("default_example.csv")
    
    # Output paths
    DATA_DIR = Path(path_directory) / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    paths = {
        'SNOMED_GRAPH_PATH': os.path.join(path_directory, SNOMED_GRAPH_RELATIVE_PATH),
        'ALL_TRANSLATIONS_PATH': os.path.join(path_directory, ALL_TRANSLATIONS_RELATIVE_PATH),
        'SAMPLES_PATH': os.path.join(path_directory, SAMPLES_RELATIVE_PATH),
        'EMBEDDINGS_DIR': os.path.join(path_directory, EMBEDDINGS_DIR_RELATIVE_PATH),
        'DEFAULT_EXAMPLE_PATH': os.path.join(path_directory, DEFAULT_EXAMPLE_RELATIVE_PATH),
        'DEFAULT_EXAMPLE_FR_ES_PATH': DATA_DIR / "default_example_fr_es.csv",
        'EXAMPLE_RETRIEVAL_PATH': DATA_DIR / "example_retrieval_es_fr.csv"
    }
    
    return paths

# Important attributes for graph relationships
IMPORTANT_ATTRIBUTES = {
    'Access (attribute)', 'After (attribute)', 'Associated finding (attribute)',
    'Associated morphology (attribute)', 'Associated procedure (attribute)',
    'Associated with (attribute)', 'Before (attribute)', 'Causative agent (attribute)',
    'Characterizes (attribute)', 'Clinical course (attribute)', 'Component (attribute)',
    'Direct device (attribute)', 'Direct morphology (attribute)', 'Direct site (attribute)',
    'Direct substance (attribute)', 'Due to (attribute)', 'During (attribute)',
    'Finding context (attribute)', 'Finding informer (attribute)', 'Finding method (attribute)',
    'Finding site (attribute)', 'Has absorbability (attribute)', 'Has active ingredient (attribute)',
    'Has basic dose form (attribute)', 'Has basis of strength substance (attribute)',
    'Has coating material (attribute)', 'Has compositional material (attribute)',
    'Has concentration strength denominator unit (attribute)', 'Has concentration strength numerator unit (attribute)',
    'Has device intended site (attribute)', 'Has disposition (attribute)',
    'Has dose form administration method (attribute)', 'Has dose form intended site (attribute)',
    'Has dose form release characteristic (attribute)', 'Has dose form transformation (attribute)',
    'Has filling (attribute)', 'Has focus (attribute)', 'Has ingredient qualitative strength (attribute)',
    'Has intent (attribute)', 'Has interpretation (attribute)', 'Has manufactured dose form (attribute)',
    'Has precise active ingredient (attribute)', 'Has presentation strength denominator unit (attribute)',
    'Has presentation strength numerator unit (attribute)', 'Has realization (attribute)',
    'Has specimen (attribute)', 'Has state of matter (attribute)', 'Has surface texture (attribute)',
    'Has target population (attribute)', 'Has unit of presentation (attribute)',
    'Indirect device (attribute)', 'Indirect morphology (attribute)', 'Inherent location (attribute)',
    'Inheres in (attribute)', 'Interprets (attribute)', 'Is a (attribute)',
    'Is modification of (attribute)', 'Is sterile (attribute)', 'Laterality (attribute)',
    'Measurement method (attribute)', 'Method (attribute)', 'Occurrence (attribute)',
    'Pathological process (attribute)', 'Plays role (attribute)', 'Precondition (attribute)',
    'Priority (attribute)', 'Procedure context (attribute)', 'Procedure device (attribute)',
    'Procedure morphology (attribute)', 'Procedure site (attribute)', 'Procedure site - Direct (attribute)',
    'Procedure site - Indirect (attribute)', 'Process acts on (attribute)', 'Process duration (attribute)',
    'Process extends to (attribute)', 'Process output (attribute)', 'Property (attribute)',
    'Recipient category (attribute)', 'Relative to (attribute)', 'Relative to part of (attribute)',
    'Revision status (attribute)', 'Route of administration (attribute)', 'Scale type (attribute)',
    'Severity (attribute)', 'Specimen procedure (attribute)', 'Specimen source identity (attribute)',
    'Specimen source morphology (attribute)', 'Specimen source topography (attribute)',
    'Specimen substance (attribute)', 'Subject relationship context (attribute)',
    'Surgical approach (attribute)', 'Technique (attribute)', 'Temporal context (attribute)',
    'Temporally related to (attribute)', 'Time aspect (attribute)', 'Units (attribute)',
    'Using access device (attribute)', 'Using device (attribute)', 'Using energy (attribute)',
    'Using substance (attribute)'
}

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def extract_clean_pt(ref_list_str):
    """Extract clean preferred term from Spanish reference translations."""
    try:
        # Convert string to Python list
        ref_list = ast.literal_eval(ref_list_str)
        # Remove duplicates
        ref_list = list(dict.fromkeys(ref_list))
        # Clean each element: remove ' [como un todo]'
        ref_list = [s.replace(' [como un todo]', '').strip() for s in ref_list]
        # Return first clean element if available
        return ref_list[0] if ref_list else None
    except Exception as e:
        # In case of bad data or format
        return None

def fix_reference_spanish(ref):
    """Fix Spanish reference translations format."""
    if isinstance(ref, str):
        try:
            ref_list = ast.literal_eval(ref)
        except Exception:
            ref_list = ref  
    else:
        ref_list = ref
    
    if isinstance(ref_list, list) and len(ref_list) > 0:
        ref_str = ref_list[0]
    else:
        ref_str = str(ref_list)
    
    # Remove "[como un todo]" substring and potential spaces before
    ref_str = re.sub(r'\s*\[como un todo\]', '', ref_str)
    return ref_str

def load_embeddings(path):
    """Load embeddings from .pkl/.pkl.gz (joblib) or .json files."""
    ext = os.path.splitext(path)[1]
    if ext in {".pkl", ".gz"}:
        d = joblib.load(path)
        return {k: (v if isinstance(v, np.ndarray) else np.asarray(v, dtype="float32"))
                for k, v in d.items()}
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: np.asarray(v, dtype="float32") for k, v in raw.items()}
    else:
        raise ValueError(f"Unknown format for {path}")

# ----------------------------------------------------------------------------
# Default example processing
# ----------------------------------------------------------------------------

def create_default_example_fr_es(paths, all_translations_df):
    """Create default_example_fr_es.csv with Spanish preferred terms added."""
    print("Creating default_example_fr_es.csv...")
    
    # Load default example file
    default_example_df = pd.read_csv(paths['DEFAULT_EXAMPLE_PATH'], sep=";", encoding="utf-8")
    print(f"Loaded {len(default_example_df)} default examples")
    
    # Clean up columns
    columns_to_drop = ['SYN FR 1', 'Présent dans set Léo', 'SYN FR 2', 'SYN FR 3', 'SYN FR 4']
    existing_columns_to_drop = [col for col in columns_to_drop if col in default_example_df.columns]
    if existing_columns_to_drop:
        default_example_df.drop(existing_columns_to_drop, axis=1, inplace=True)
    
    # Rename columns
    if 'Détail' in default_example_df.columns:
        default_example_df.rename(columns={'Détail': 'detail'}, inplace=True)
    
    # Normalize hierarchy column
    if 'Hiérarchie' in default_example_df.columns:
        default_example_df.rename(columns={'Hiérarchie': 'hierarchy'}, inplace=True)
    
    for index, row in default_example_df.iterrows():
        if 'hierarchy' in row:
            hier = row['hierarchy']
            hier_case = hier.casefold()
            default_example_df.at[index, 'hierarchy'] = hier_case
    
    # Filter Spanish translations
    es_tran_df = all_translations_df[all_translations_df['language'] == 'Spanish'].copy()
    
    # Clean Spanish translations
    es_tran_df['clean_pt_es'] = es_tran_df['reference_translations'].apply(extract_clean_pt)
    
    # Create mapping dictionary
    mapping_dict = {
        (row.sctid, row.hierarchy): row.clean_pt_es
        for _, row in es_tran_df.iterrows()
    }
    
    # Apply mapping to get Spanish preferred terms
    def get_pt_es(row):
        if 'SCTID' in row and 'hierarchy' in row:
            key = (row['SCTID'], row['hierarchy'])
        elif 'sctid' in row and 'hierarchy' in row:
            key = (row['sctid'], row['hierarchy'])
        else:
            return None
        return mapping_dict.get(key, None)
    
    default_example_df['pt_es'] = default_example_df.apply(get_pt_es, axis=1)
    
    # Save result
    default_example_df.to_csv(paths['DEFAULT_EXAMPLE_FR_ES_PATH'], sep="\t", index=False)
    print(f"Saved default_example_fr_es.csv to {paths['DEFAULT_EXAMPLE_FR_ES_PATH']}")
    
    return default_example_df

# ----------------------------------------------------------------------------
# Graph-based retrieval functions
# ----------------------------------------------------------------------------

def get_ancestors_with_degree(G, start_sctid, root_sctid="138875005"):
    """
    Return list of tuples (ancestor_sctid, degree),
    where degree represents the distance (number of levels)
    separating start_sctid from the ancestor.
    The root is not included.
    """
    visited = set()
    queue = []
    ancestors_with_degree = []

    # Get immediate parents
    start_concept = G.get_full_concept(start_sctid)
    # Direct parents are at distance = 1
    for p in start_concept.parents:
        queue.append((p.sctid, 1))

    while queue:
        current_sctid, dist = queue.pop(0)
        # Don't include the root
        if current_sctid == root_sctid:
            continue

        # Avoid re-visits
        if current_sctid in visited:
            continue
        visited.add(current_sctid)

        # Add this ancestor to the list
        ancestors_with_degree.append((current_sctid, dist))

        # Get its parents to continue
        current_parents = G.get_full_concept(current_sctid).parents
        for cp in current_parents:
            queue.append((cp.sctid, dist + 1))

    return ancestors_with_degree

def collect_graph_examples(concept_row, G, all_translations_df):
    """
    Collect graph-based examples for a concept including:
    1) Parents (immediate)
    2) Siblings (same parent)
    3) Concepts linked by important attributes
    4) Ancestors with degree of parenthood
    """
    sctid = concept_row["sctid"]
    language = concept_row["language"]
    concept = G.get_full_concept(sctid)
    
    # Create index for fast lookup
    translations_idx = all_translations_df.set_index(['sctid', 'language'])
    
    examples = []
    
    # 1) Parents
    parent_concepts = concept.parents
    for parent in parent_concepts:
        if (parent.sctid, language) in translations_idx.index:
            parent_row = translations_idx.loc[(parent.sctid, language)]
            if parent_row.has_translation:
                parent_full = G.get_full_concept(parent.sctid)
                parent_pt = parent_full.fsn.replace(f"({parent_full.hierarchy})", "").strip()
                examples.append({
                    "source_sctid": sctid,
                    "source_language": language,
                    "example_sctid": parent.sctid,
                    "example_preferred_term": parent_pt,
                    "example_translation": fix_reference_spanish(parent_row.reference_translations) if language == 'Spanish' else parent_row.reference_translations,
                    "retrieval_method": "graph",
                    "relation_type": "parent",
                    "score": 1.0,
                    "degree_of_parenthood": 1
                })
    
    # 2) Siblings
    siblings = set()
    for parent in parent_concepts:
        siblings.update(G.get_full_concept(parent.sctid).children)
    siblings = siblings - {concept}  # Remove the concept itself
    
    for sibling in siblings:
        if (sibling.sctid, language) in translations_idx.index:
            sibling_row = translations_idx.loc[(sibling.sctid, language)]
            if sibling_row.has_translation:
                sibling_full = G.get_full_concept(sibling.sctid)
                sibling_pt = sibling_full.fsn.replace(f"({sibling_full.hierarchy})", "").strip()
                examples.append({
                    "source_sctid": sctid,
                    "source_language": language,
                    "example_sctid": sibling.sctid,
                    "example_preferred_term": sibling_pt,
                    "example_translation": fix_reference_spanish(sibling_row.reference_translations) if language == 'Spanish' else sibling_row.reference_translations,
                    "retrieval_method": "graph",
                    "relation_type": "sibling",
                    "score": 0.9,
                    "degree_of_parenthood": None
                })
    
    # 3) Attribute relationships
    for group in concept.inferred_relationship_groups:
        for rel in group.relationships:
            if rel.type in IMPORTANT_ATTRIBUTES:
                if (rel.tgt.sctid, language) in translations_idx.index:
                    rel_row = translations_idx.loc[(rel.tgt.sctid, language)]
                    if rel_row.has_translation:
                        related_concept = G.get_full_concept(rel.tgt.sctid)
                        related_pt = related_concept.fsn.replace(f"({related_concept.hierarchy})", "").strip()
                        examples.append({
                            "source_sctid": sctid,
                            "source_language": language,
                            "example_sctid": rel.tgt.sctid,
                            "example_preferred_term": related_pt,
                            "example_translation": fix_reference_spanish(rel_row.reference_translations) if language == 'Spanish' else rel_row.reference_translations,
                            "retrieval_method": "graph",
                            "relation_type": "attribute",
                            "exact_relation": rel.type,
                            "score": 0.8,
                            "degree_of_parenthood": None
                        })
    
    # 4) Ancestors
    ancestors_with_dist = get_ancestors_with_degree(G, sctid)
    for anc_sctid, dist in ancestors_with_dist:
        if (anc_sctid, language) in translations_idx.index:
            anc_row = translations_idx.loc[(anc_sctid, language)]
            if anc_row.has_translation:
                anc_concept = G.get_full_concept(anc_sctid)
                anc_pt = anc_concept.fsn.replace(f"({anc_concept.hierarchy})", "").strip()
                examples.append({
                    "source_sctid": sctid,
                    "source_language": language,
                    "example_sctid": anc_sctid,
                    "example_preferred_term": anc_pt,
                    "example_translation": fix_reference_spanish(anc_row.reference_translations) if language == 'Spanish' else anc_row.reference_translations,
                    "retrieval_method": "graph",
                    "relation_type": "ancestor",
                    "score": max(0.1, 0.7 - (dist * 0.1)),  # Decreasing score with distance
                    "degree_of_parenthood": dist
                })
    
    return examples

# ----------------------------------------------------------------------------
# Similarity-based retrieval functions
# ----------------------------------------------------------------------------

def setup_embeddings(embeddings_dir):
    """Setup embeddings for similarity search."""
    print("Loading embeddings...")
    
    # Load embeddings
    emb_graph = load_embeddings(os.path.join(embeddings_dir, "graph_node2vec_high_q.pkl.gz"))
    emb_bow = load_embeddings(os.path.join(embeddings_dir, "bow_binary_ngram.pkl.gz"))
    emb_tfidf = load_embeddings(os.path.join(embeddings_dir, "tfidf_ngram.pkl.gz"))
    emb_encoder = load_embeddings(os.path.join(embeddings_dir, "st_multilingual.pkl.gz"))
    
    EMBEDS = {
        "graph": emb_graph,
        "bow": emb_bow,
        "tfidf": emb_tfidf,
        "encoder": emb_encoder
    }
    
    # Convert keys to int
    EMBEDS = {n: {int(k): v for k, v in d.items()} for n, d in EMBEDS.items()}
    
    return EMBEDS

def setup_similarity_indices(EMBEDS):
    """Setup GPU and Faiss indices for similarity search."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # GPU embeddings (graph, encoder)
    GPU_EMB, GPU_ID = {}, {}
    
    for name in ("graph", "encoder"):
        if name in EMBEDS:
            ids, vecs = zip(*EMBEDS[name].items())
            arr = np.vstack(vecs).astype(np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            mat = torch.from_numpy(arr).to(DEVICE)
            GPU_EMB[name] = mat.half()
            GPU_ID[name] = torch.tensor(ids, device=DEVICE, dtype=torch.int64)
            del arr
    
    # Faiss indices (bow, tfidf)
    FAISS = {}
    
    for name in ("bow", "tfidf"):
        if name in EMBEDS:
            ids, vecs = zip(*EMBEDS[name].items())
            arr = np.vstack(vecs).astype('float32')
            arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            index = faiss.IndexFlatIP(arr.shape[1])
            index.add(arr)
            id_arr = np.asarray(ids, dtype='int64')
            FAISS[name] = (index, id_arr)
            del arr
    
    return GPU_EMB, GPU_ID, FAISS, DEVICE

def topk_gpu(src_vec: np.ndarray, sim_type: str, excluded: Set[int], 
             k: int, min_score: float, GPU_EMB, GPU_ID, DEVICE) -> List[Tuple[int, float]]:
    """Find top-k similar concepts using GPU."""
    src = torch.from_numpy(src_vec.astype('float32')).to(DEVICE)
    src = F.normalize(src, dim=0, eps=1e-9).half()
    sims = (GPU_EMB[sim_type] @ src).float()
    
    if excluded:
        mask = torch.isin(GPU_ID[sim_type], torch.tensor(list(excluded), device=DEVICE))
        sims[mask] = -1.0
    
    topv, topi = torch.topk(sims, k=k*2)
    return [(int(GPU_ID[sim_type][i]), float(topv[j])) 
            for j, i in enumerate(topi) if topv[j] >= min_score][:k]

def topk_faiss(src_vec: np.ndarray, sim_type: str, excluded: Set[int], 
               k: int, min_score: float, FAISS) -> List[Tuple[int, float]]:
    """Find top-k similar concepts using Faiss."""
    src = src_vec.astype('float32', copy=False)
    src /= np.linalg.norm(src) + 1e-9
    index, id_arr = FAISS[sim_type]
    D, I = index.search(src[None, :], k*2)
    
    res = []
    for score, idx in zip(D[0], I[0]):
        cid = int(id_arr[idx])
        if cid in excluded or score < min_score:
            continue
        res.append((cid, float(score)))
        if len(res) == k:
            break
    return res

def collect_similarity_examples(concept_row, G, all_translations_df, EMBEDS, 
                              GPU_EMB, GPU_ID, FAISS, DEVICE, k=5, min_score=0.1):
    """Collect similarity-based examples for a concept."""
    sctid = int(concept_row['sctid'])
    language = concept_row['language']
    concept = G.get_full_concept(sctid)
    
    # Create index for fast lookup
    translations_idx = all_translations_df.set_index(['sctid', 'language'])
    
    # Concepts to exclude (self, descendants, parents)
    excluded = {sctid}
    excluded |= {c.sctid for c in G.get_descendants(sctid)}
    excluded |= {c.sctid for c in G.get_parents(sctid)}
    
    examples = []
    
    for sim_type in EMBEDS.keys():
        if sctid not in EMBEDS[sim_type]:
            continue
        
        src_vec = EMBEDS[sim_type][sctid]
        
        if sim_type in GPU_EMB:
            topk = topk_gpu(src_vec, sim_type, excluded, k, min_score, GPU_EMB, GPU_ID, DEVICE)
        else:
            topk = topk_faiss(src_vec, sim_type, excluded, k, min_score, FAISS)
        
        for cid, score in topk:
            if (cid, language) not in translations_idx.index:
                continue
            
            cid_row = translations_idx.loc[(cid, language)]
            if not cid_row.has_translation:
                continue
            
            tgt = G.get_full_concept(cid)
            examples.append({
                "source_sctid": sctid,
                "source_language": language,
                "example_sctid": cid,
                "example_preferred_term": tgt.fsn.replace(f"({tgt.hierarchy})", "").strip(),
                "example_translation": fix_reference_spanish(cid_row.reference_translations) if language == 'Spanish' else cid_row.reference_translations,
                "retrieval_method": "similarity",
                "similarity_type": sim_type,
                "score": round(score, 4),
                "degree_of_parenthood": None
            })
    
    return examples

# ----------------------------------------------------------------------------
# Main processing function
# ----------------------------------------------------------------------------

def process_example_retrieval(paths):
    """Main function to process example retrieval."""
    print("=" * 80)
    print("SNOMED CT Example Retrieval Processing")
    print("=" * 80)
    
    # Load SNOMED CT graph
    print("Loading SNOMED CT graph...")
    G = SnomedGraph.from_serialized(paths['SNOMED_GRAPH_PATH'])
    print(f"Loaded graph with {len(G):,} concepts")
    
    # Load data files
    print("Loading data files...")
    all_translations_df = pd.read_csv(paths['ALL_TRANSLATIONS_PATH'])
    samples_df = pd.read_csv(paths['SAMPLES_PATH'])
    
    print(f"Loaded {len(all_translations_df):,} translation records")
    print(f"Loaded {len(samples_df):,} sample concepts")
    
    # Create default_example_fr_es.csv
    create_default_example_fr_es(paths, all_translations_df)
    
    # Filter translations to those that have translations
    translations_with_trans = all_translations_df[all_translations_df['has_translation'] == True].copy()
    
    # Fix Spanish translations format
    spanish_mask = translations_with_trans['language'] == 'Spanish'
    translations_with_trans.loc[spanish_mask, 'reference_translations'] = \
        translations_with_trans.loc[spanish_mask, 'reference_translations'].apply(fix_reference_spanish)
    
    print(f"Processing {len(translations_with_trans):,} concepts with translations")
    
    # Setup embeddings for similarity search
    EMBEDS = setup_embeddings(paths['EMBEDDINGS_DIR'])
    GPU_EMB, GPU_ID, FAISS, DEVICE = setup_similarity_indices(EMBEDS)
    
    # Process examples from samples
    print("Processing sample concepts...")
    all_examples = []
    
    for _, sample_row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Processing samples"):
        # Get graph-based examples
        graph_examples = collect_graph_examples(sample_row, G, translations_with_trans)
        all_examples.extend(graph_examples)
        
        # Get similarity-based examples
        similarity_examples = collect_similarity_examples(
            sample_row, G, translations_with_trans, EMBEDS, 
            GPU_EMB, GPU_ID, FAISS, DEVICE, k=5, min_score=0.1
        )
        all_examples.extend(similarity_examples)
    
    # Convert to DataFrame
    print("Creating final DataFrame...")
    examples_df = pd.DataFrame(all_examples)
    
    # Add source concept information
    source_info = samples_df.set_index(['sctid', 'language'])
    examples_df = examples_df.merge(
        source_info[['fsn', 'hierarchy', 'reference_translations']],
        left_on=['source_sctid', 'source_language'],
        right_index=True,
        suffixes=('', '_source')
    )
    
    # Rename columns for clarity
    examples_df = examples_df.rename(columns={
        'fsn': 'source_fsn',
        'hierarchy': 'source_hierarchy',
        'reference_translations': 'source_translation'
    })
    
    # Fix source translations format
    spanish_source_mask = examples_df['source_language'] == 'Spanish'
    examples_df.loc[spanish_source_mask, 'source_translation'] = \
        examples_df.loc[spanish_source_mask, 'source_translation'].apply(fix_reference_spanish)
    
    # Sort by source concept and score
    examples_df = examples_df.sort_values([
        'source_sctid', 'source_language', 'retrieval_method', 'score'
    ], ascending=[True, True, True, False])
    
    # Save final result
    print(f"Saving example retrieval results to {paths['EXAMPLE_RETRIEVAL_PATH']}")
    examples_df.to_csv(paths['EXAMPLE_RETRIEVAL_PATH'], sep='\t', index=False)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total examples retrieved: {len(examples_df):,}")
    print(f"Source concepts processed: {examples_df['source_sctid'].nunique():,}")
    print(f"Languages: {examples_df['source_language'].unique()}")
    print(f"Retrieval methods: {examples_df['retrieval_method'].unique()}")
    
    method_counts = examples_df['retrieval_method'].value_counts()
    for method, count in method_counts.items():
        print(f"  {method}: {count:,} examples")
    
    if 'similarity_type' in examples_df.columns:
        print(f"Similarity types: {examples_df['similarity_type'].dropna().unique()}")
    
    if 'relation_type' in examples_df.columns:
        print(f"Graph relation types: {examples_df['relation_type'].dropna().unique()}")
    
    print("\n🎉 Example retrieval completed successfully!")
    
    return examples_df

# ----------------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------------

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Process SNOMED CT example retrieval")
    args = parser.parse_args()
    
    # Setup paths
    paths = setup_paths()
    
    # Check if required files exist
    required_files = [
        'SNOMED_GRAPH_PATH',
        'ALL_TRANSLATIONS_PATH', 
        'SAMPLES_PATH',
        'DEFAULT_EXAMPLE_PATH'
    ]
    
    for file_key in required_files:
        if not os.path.exists(paths[file_key]):
            print(f"Error: Required file not found: {paths[file_key]}")
            return 1
    
    # Check embeddings directory
    embeddings_files = [
        'bow_binary_ngram.pkl.gz',
        'graph_node2vec_high_q.pkl.gz', 
        'st_multilingual.pkl.gz',
        'tfidf_ngram.pkl.gz'
    ]
    
    for emb_file in embeddings_files:
        emb_path = os.path.join(paths['EMBEDDINGS_DIR'], emb_file)
        if not os.path.exists(emb_path):
            print(f"Error: Required embedding file not found: {emb_path}")
            return 1
    
    try:
        # Process example retrieval
        examples_df = process_example_retrieval(paths)
        
        print(f"\nFiles created:")
        print(f"- {paths['DEFAULT_EXAMPLE_FR_ES_PATH']}")
        print(f"- {paths['EXAMPLE_RETRIEVAL_PATH']}")
        
        return 0
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
