
"""
load_translations_and_embedding.py – Load SNOMED CT translations and generate embeddings
====================================================================================

This script loads SNOMED CT translation data from French and Spanish extensions,
creates the all_translations.csv file with comprehensive concept analysis,
and generates all required embedding spaces for the translation retrieval system.

The script performs the following steps:
1. Load SNOMED CT graph and translation data
2. Calculate context and similarity tiers for concepts
3. Generate all_translations.csv with concept metadata
4. Create samples.csv for experimentation
5. Generate all embedding spaces (BoW, TF-IDF, SentenceTransformer)

Author: L. Megret – Inria SED (2025)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, Set, List, Any
from itertools import groupby
import json
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from scipy.sparse import lil_matrix, csc_matrix
from ast import literal_eval

# NLP and ML
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer


# Graph processing
import networkx as nx
from snomed_graph.snomed_graph import SnomedGraph

# Utility function for chunking
def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------

def setup_paths():
    """Setup all file paths for the project."""
    path_directory = os.getcwd()
    
    # Input paths
    FRENCH_TRAD_VAL_RELATIVE_PATH = os.path.join("SnomedCT_FR", "traductions_validées_20241218_v2.csv")
    SPANISH_REFSET_RELATIVE_PATH = os.path.join("SnomedCT_SpanishRelease-es_PRODUCTION_20240930T120000Z", "Snapshot", "Refset", "Language", "der2_cRefset_LanguageSpanishExtensionSnapshot-es_INT_20240930.txt")
    SPANISH_DESCRIPTION_RELATIVE_PATH = os.path.join("SnomedCT_SpanishRelease-es_PRODUCTION_20240930T120000Z", "Full", "Terminology", "sct2_Description_SpanishExtensionFull-es_INT_20240930.txt")
    SNOMED_GRAPH_RELATIVE_PATH = os.path.join("snomed_graph", "full_concept_graph_snomed_ct_int_rf2_20241201.gml")
    
    # Output paths
    DATA_DIR = Path(path_directory) / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    paths = {
        'FRENCH_TRAD_VAL_PATH': os.path.join(path_directory, FRENCH_TRAD_VAL_RELATIVE_PATH),
        'SPANISH_REFSET_PATH': os.path.join(path_directory, SPANISH_REFSET_RELATIVE_PATH),
        'SPANISH_DESCRIPTION_PATH': os.path.join(path_directory, SPANISH_DESCRIPTION_RELATIVE_PATH),
        'SNOMED_GRAPH_PATH': os.path.join(path_directory, SNOMED_GRAPH_RELATIVE_PATH),
        'ALL_TRANSLATIONS_PATH': DATA_DIR / "all_translations.csv",
        'SAMPLE_PATH': DATA_DIR / "samples.csv",
        'EMBEDDINGS_DIR': EMBEDDINGS_DIR
    }
    
    return paths

# Hierarchies and attributes of interest
HIERARCHIES_IN_USE = [
    "substance",
    "body structure", 
    "finding",
    "disorder",
    "procedure"
]

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
# File existence checking functions
# ----------------------------------------------------------------------------

def check_existing_files(paths):
    """Check which files already exist and return status."""
    existing_files = {}
    existing_files['all_translations'] = paths['ALL_TRANSLATIONS_PATH'].exists()
    existing_files['samples'] = paths['SAMPLE_PATH'].exists()
    
    # Check embedding files (only the .pkl.gz files are needed)
    embedding_files = [
        'bow_binary_ngram.pkl.gz',
        'tfidf_ngram.pkl.gz',
        'st_multilingual.pkl.gz'
    ]
    
    existing_files['embeddings'] = {}
    for emb_file in embedding_files:
        existing_files['embeddings'][emb_file] = (paths['EMBEDDINGS_DIR'] / emb_file).exists()
    
    return existing_files

def print_file_status(existing_files, paths):
    """Print status of existing files."""
    print("\nFile Status Check:")
    print("-" * 40)
    
    status_symbol = lambda exists: "✅" if exists else "❌"
    
    print(f"{status_symbol(existing_files['all_translations'])} all_translations.csv")
    print(f"{status_symbol(existing_files['samples'])} samples.csv")
    
    print("\nEmbedding Files:")
    for emb_file, exists in existing_files['embeddings'].items():
        print(f"  {status_symbol(exists)} {emb_file}")

# ----------------------------------------------------------------------------
# Translation loading functions
# ----------------------------------------------------------------------------

def load_translation_spanish(G: SnomedGraph, desc_path: str, lang_path: str) -> pd.DataFrame:
    """Load Spanish translations from SNOMED CT Spanish extension."""
    print("Loading Spanish translations...")
    
    # Load the concept descriptions
    desc_df = pd.read_csv(desc_path, delimiter="\t", encoding='utf-8')
    # Load the language refset
    lang_df = pd.read_csv(lang_path, delimiter="\t", encoding='utf-8')
    
    # Filter the refset to Preferred Terms only
    lang_df = lang_df[lang_df.acceptabilityId == 900000000000548007]
    # IDs of all descriptors which are preferred terms
    preferred_term_descriptor_ids = lang_df.referencedComponentId.unique()
    
    # Filter descriptions to active concepts only
    desc_df = desc_df[desc_df.active == 1]
    # Filter to preferred terms
    desc_df = desc_df[desc_df.id.isin(preferred_term_descriptor_ids)]
    # Remove FSNs
    desc_df = desc_df[desc_df.typeId != 900000000000003001]
    # Some extensions include English terms. We don't want these.
    desc_df = desc_df[desc_df.languageCode != "en"]
    
    # Remove concepts that don't exist in the International Edition
    desc_df = desc_df[[sctid in G for sctid in desc_df.conceptId]]
    desc_df = desc_df.rename(axis="columns", mapper={"conceptId": "sctid"})
    
    # One row per concept, with the synonyms aggregated into a list
    desc_df = desc_df.groupby("sctid").term.apply(list).rename("translations").to_frame()
    
    print(f"Loaded {len(desc_df)} Spanish translations")
    return desc_df

def load_translation_french(G: SnomedGraph, trans_path: str) -> pd.DataFrame:
    """Load French translations from validated translation file."""
    print("Loading French translations...")
    
    # Load the concept translations
    trans_df = pd.read_csv(trans_path, delimiter="\t", encoding='utf-8')
    # Filter to preferred terms
    trans_df = trans_df[trans_df.acceptability == 'PT']
    
    # Remove concepts that don't exist in the International Edition
    trans_df = trans_df[[sctid in G for sctid in trans_df.sctid]]
    trans_df = trans_df.rename(axis="columns", mapper={"descriptionId": "id"})
    
    # One row per concept, with the synonyms aggregated into a list
    trans_df = trans_df.groupby("sctid").term.apply(list).rename("translations").to_frame()
    
    print(f"Loaded {len(trans_df)} French translations")
    return trans_df

# ----------------------------------------------------------------------------
# Context tier calculation
# ----------------------------------------------------------------------------

def calc_context_tiers(langcode: str, translations: Dict[str, pd.DataFrame], G: SnomedGraph):
    """Calculate context tiers for concepts based on translation availability."""
    print(f"Calculating context tiers for {langcode}...")
    
    tier_0_concepts = set([c.sctid for c in G])
    all_translations = set(translations[langcode].index.tolist())

    print("  Calculating Context Tier 1 Concept Set")
    # Tier 1 concepts are concepts where all parents have also been translated
    tier_1_concepts = set([
        c for c in tqdm(tier_0_concepts, desc="Tier 1")
        if all([
            p.sctid in all_translations
            for p in G.get_full_concept(c).parents
        ])
    ])

    print("  Calculating Context Tier 2 Concept Set")
    # Tier 2 concepts are Tier 1 concepts where important defining attributes have also been translated
    tier_2_concepts = set([
        c for c in tqdm(tier_1_concepts, desc="Tier 2")
        if all([
            r.tgt.sctid in all_translations
            for g in G.get_full_concept(c).inferred_relationship_groups
            for r in g.relationships            
        ]) 
        and len(G.get_full_concept(c).inferred_relationship_groups) > 0
    ])

    tier_0_concepts = tier_0_concepts - tier_1_concepts - tier_2_concepts
    tier_1_concepts = tier_1_concepts - tier_2_concepts

    print(f"""    Language: {langcode}
    Tier 0: {len(tier_0_concepts):,}
    Tier 1: {len(tier_1_concepts):,}
    Tier 2: {len(tier_2_concepts):,}""")

    return tier_0_concepts, tier_1_concepts, tier_2_concepts

# ----------------------------------------------------------------------------
# Depth tier calculation
# ----------------------------------------------------------------------------

def calc_depth_tiers(G: SnomedGraph):
    """Calculate depth tiers based on distance from root concepts."""
    print("Calculating depth tiers...")
    
    shallow_tier = set()
    mid_tier = set()
    deep_tier = set()
    
    for concept in tqdm(iter(G), total=len(G), desc="Calculating depths"):
        try:
            depth = len(G.path_to_root(concept.sctid))
        except TypeError:
            pass
        else:
            if 1 <= depth <= 4:
                shallow_tier.add(concept.sctid)
            elif 5 <= depth <= 7:
                mid_tier.add(concept.sctid)
            elif depth >= 8:
                deep_tier.add(concept.sctid)
    
    print(f"  Shallow: {len(shallow_tier):,}, Mid: {len(mid_tier):,}, Deep: {len(deep_tier):,}")
    return shallow_tier, mid_tier, deep_tier

# ----------------------------------------------------------------------------
# Similarity tier calculation
# ----------------------------------------------------------------------------

def sparse_to_torch(sparse_matrix):
    """Convert scipy sparse matrix to torch sparse tensor."""
    coo = sparse_matrix.tocoo()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)

def calc_similarity_tiers(translations: Dict[str, pd.DataFrame], G: SnomedGraph, min_score=2, chunksize=1000):
    """Calculate similarity tiers based on n-gram overlap between concept terms."""
    print("Calculating similarity tiers...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    similarity_tiers = {}
    tier_0_concepts = [c.sctid for c in G]
    candidate_tier_1_concepts = {}
    preferred_terms = [
        c.fsn.replace(f"({c.hierarchy})", "").strip() for c in G
    ]
    
    print("  Vectorizing preferred terms...")
    vectorizer = CountVectorizer(
        lowercase=True, stop_words=None, ngram_range=(2, 10), binary=True
    )
    key_matrix = vectorizer.fit_transform(preferred_terms)
    
    # Convert to sparse tensor on GPU if available
    key_matrix_torch = sparse_to_torch(key_matrix)

    print("  Finding similar terms...")    
    N_iter = max(1, len(tier_0_concepts) // chunksize)
    it = zip(chunked(tier_0_concepts, chunksize), chunked(preferred_terms, chunksize))
    
    for sctids, pt_chunk in tqdm(it, total=N_iter, desc="Finding similarities"):
        queries = vectorizer.transform(pt_chunk)
        queries_torch = sparse_to_torch(queries)
        
        # Optimized matrix multiplication with sparse tensors
        search_torch = torch.sparse.mm(queries_torch, key_matrix_torch.T)
        search = search_torch.to_dense().cpu().numpy()

        similar = lil_matrix(search >= min_score)
        src_idx, tgt_idx = similar.nonzero()

        it2 = groupby(zip(src_idx, tgt_idx), key=lambda x: x[0])
        for src, grp in it2:
            src_sctid = sctids[src]
            tgt_sctids = {tier_0_concepts[tgt] for _, tgt in grp}
            candidate_tier_1_concepts[src_sctid] = tgt_sctids

    print("  Filtering similar terms...")
    for sctid in tqdm(candidate_tier_1_concepts.keys(), desc="Filtering"):
        descendants = {c.sctid for c in G.get_descendants(sctid)}
        parents = {c.sctid for c in G.get_parents(sctid)}
        candidate_tier_1_concepts[sctid] -= {sctid} | descendants | parents

    print("  Filtering by language...")
    for langcode, translations_df in tqdm(translations.items(), desc="Languages"):
        all_translations = set(translations_df.index.tolist())
        tier_1_concepts = {
            sctid
            for sctid, others in candidate_tier_1_concepts.items()
            if others & all_translations
        }
        
        remaining_tier_0_concepts = set(tier_0_concepts) - tier_1_concepts
        
        similarity_tiers[langcode] = {
            "tier0": remaining_tier_0_concepts, 
            "tier1": tier_1_concepts
        }

        print(f"""    Language: {langcode}
    Tier 0: {len(remaining_tier_0_concepts):,}
    Tier 1: {len(tier_1_concepts):,}""")

    return similarity_tiers

# ----------------------------------------------------------------------------
# DataFrame generation
# ----------------------------------------------------------------------------

def generate_all_concepts_df(G: SnomedGraph, translations: Dict[str, pd.DataFrame], 
                           context_tiers: Dict, similarity_tiers: Dict, 
                           depth_tiers: tuple):
    """Generate comprehensive DataFrame with all concept metadata."""
    print("Generating comprehensive concepts DataFrame...")
    
    ts, tm, td = depth_tiers
    languages = list(translations.keys())
    
    def get_concept_len_bucket(concept):
        preferred_term = concept.fsn.replace(f"({concept.hierarchy})", "").strip()
        if len(preferred_term) <= 20:
            return "Short"
        elif len(preferred_term) <= 30:
            return "Medium"
        else:
            return "Long"
            
    def get_depth(sctid):
        if sctid in ts:
            return "Shallow"
        elif sctid in tm:
            return "Medium"
        elif sctid in td:
            return "Deep"
        else:
            return pd.NA
            
    def get_cxt_tier(sctid, lang):
        if sctid in context_tiers[lang]["tier2"]:
            return "Tier 2"
        elif sctid in context_tiers[lang]["tier1"]:
            return "Tier 1"
        elif sctid in context_tiers[lang]["tier0"]:
            return "Tier 0"
        else:
            return pd.NA
            
    def get_sim_tier(sctid, lang):
        if sctid in similarity_tiers[lang]["tier1"]:
            return "Tier 1"
        else:
            return "Tier 0"
    
    data = []
    for concept in tqdm(iter(G), total=len(G), desc="Processing concepts"):
        for lang in languages:
            try:
                translated_synonyms = translations[lang].loc[concept.sctid].translations
            except KeyError:
                translated_synonyms = pd.NA
                
            data.append({
                'sctid': concept.sctid,
                'fsn': concept.fsn,
                'hierarchy': concept.hierarchy,
                'depth_tier': get_depth(concept.sctid),
                'language': lang,
                'context_tier': get_cxt_tier(concept.sctid, lang),
                'similarity_tier': get_sim_tier(concept.sctid, lang),
                'concept_length_bucket': get_concept_len_bucket(concept),
                'reference_translations': translated_synonyms,
            })
    
    df = pd.DataFrame(data)
    df["has_translation"] = df.reference_translations.apply(lambda x: True if isinstance(x, list) else False)
    
    return df

def create_sample_df(df: pd.DataFrame, sample_size: int = 25):
    """Create sample DataFrame for experimentation."""
    print("Creating sample DataFrame...")
    
    def sample_group(grp, sample_size=sample_size):
        sample_size = min(grp.shape[0], sample_size)
        sample = grp.sample(sample_size, replace=False)
        return sample[["sctid", "fsn", "reference_translations"]]

    sample_df = (
        df[
            (df.hierarchy.isin(HIERARCHIES_IN_USE)) &
            (df.has_translation)
        ]
        .dropna()
        .groupby(["hierarchy", "depth_tier", "language", "context_tier", "similarity_tier", "concept_length_bucket"])
        .apply(sample_group)
        .reset_index()
        .drop("level_6", axis="columns")
        .sort_values(["language", "hierarchy", "depth_tier", "context_tier", "similarity_tier", "concept_length_bucket"])
    )
    
    return sample_df

# ----------------------------------------------------------------------------
# Embedding generation functions
# ----------------------------------------------------------------------------
def build_bow_embeddings_json(df,
                            ngram_range=(2, 10),
                            fname="bow_binary_ngram"):
    """
    Construit un dictionnaire {sctid: vector} BoW (binaire n‑grammes)
    puis le sérialise en pickle compressé (.pkl.gz).  
    Optionnellement, un JSON + barre de progression est écrit via
    _save_json_progress(dump_json=True).
    """
    sentences = df["fsn"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    sctids    = df["sctid"].astype(str).tolist()

    vect = CountVectorizer(lowercase=True, binary=True, ngram_range=ngram_range)
    X = vect.fit_transform(sentences)        # sparse matrix (n_samples, n_features)

    emb_dct = {}
    for i, cid in tqdm(enumerate(sctids),
                    total=len(sctids),
                    desc=f"BoW n‑gram vectors ({fname})",
                    unit="vec"):
        emb_dct[cid] = X[i].toarray().ravel()  # ➜ numpy array (plus compact)

    # ── sauvegardes ──────────────────────────────────────────────────────
    os.makedirs(VEC_DIR, exist_ok=True)

    # 1. pickle compressé (recommandé pour usage interne)
    pkl_path = f"{VEC_DIR}/{fname}.pkl.gz"
    joblib.dump(emb_dct, pkl_path, compress=("gzip", 3))
    joblib.dump(vect, f"{VEC_DIR}/{fname}_vectorizer.joblib")
    print(f"✅ Embeddings BoW picklés dans {pkl_path}")

def build_tfidf_embeddings_json(df, ngram_range=(1, 3),
                                fname="tfidf_ngram"):
    """
    Sérialise un dict {sctid: np.ndarray<float32>} de TF‑IDF et le vectorizer.
    pickle(.pkl.gz) ≈ 10× plus rapide et plus compact que JSON.
    """
    sentences = df["fsn"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    sctids    = df["sctid"].astype(str).tolist()

    vect = TfidfVectorizer(lowercase=True, ngram_range=ngram_range)
    X = vect.fit_transform(sentences).astype("float32")   # sparse CSR

    emb_dct = {}
    for i, cid in tqdm(enumerate(sctids), total=len(sctids),
                    desc=f"TF‑IDF vectors ({fname})", unit="vec"):
        emb_dct[cid] = X[i].toarray().ravel()

    # --- sauvegardes -------------------------------------------------------
    os.makedirs(VEC_DIR, exist_ok=True)
    pkl_path = f"{VEC_DIR}/{fname}.pkl.gz"
    joblib.dump(emb_dct, pkl_path, compress=("gzip", 3))
    joblib.dump(vect, f"{VEC_DIR}/{fname}_vectorizer.joblib")
    print(f"✅ Embeddings TF‑IDF picklés dans {pkl_path}")

def build_encoder_embeddings_json(df,
                                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                batch_size=128,
                                fname="st_multilingual"):
    """
    Encode les FSN → embeddings  (normalisés).  Stockage principal : pickle gzip.
    """
    sentences = df["fsn"].str.replace(r"\(.*\)", "", regex=True).str.strip().tolist()
    sctids    = df["sctid"].astype(str).tolist()

    model  = SentenceTransformer(model_name)

    embeds = model.encode(sentences,
                        batch_size=batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=True).astype("float32")

    emb_dct = {}
    for cid, vec in tqdm(zip(sctids, embeds), total=len(sctids),
                        desc=f"ST vectors ({fname})", unit="vec"):
        emb_dct[cid] = vec        # np.ndarray 384/512 dims

    # --- sauvegardes -------------------------------------------------------
    os.makedirs(VEC_DIR, exist_ok=True)
    pkl_path = f"{VEC_DIR}/{fname}.pkl.gz"
    joblib.dump(emb_dct, pkl_path, compress=("gzip", 3))
    print(f"✅ Embeddings ST picklés dans {pkl_path}")
        
# ----------------------------------------------------------------------------
# Main execution function
# ----------------------------------------------------------------------------

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Load SNOMED CT translations and generate embeddings")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all files, even if they exist")
    args = parser.parse_args()
    
    # Setup paths
    paths = setup_paths()
    
    print("=" * 80)
    print("SNOMED CT Translation Loading and Embedding Generation")
    print("=" * 80)
    
    # Check existing files
    existing_files = check_existing_files(paths)
    print_file_status(existing_files, paths)
    
    # Determine what needs to be computed
    need_translations_data = not existing_files['all_translations'] or not existing_files['samples'] or args.force
    need_embeddings = not args.skip_embeddings and (not all(existing_files['embeddings'].values()) or args.force)
    
    if not need_translations_data and not need_embeddings:
        print("\n🎉 All files already exist! Use --force to regenerate.")
        return
    
    # Load SNOMED CT graph (needed for both translations and embeddings)
    print("\nLoading SNOMED CT graph...")
    G = SnomedGraph.from_serialized(paths['SNOMED_GRAPH_PATH'])
    print(f"Loaded graph with {len(G):,} concepts")
    
    # Only compute translations data if needed
    if need_translations_data:
        if existing_files['all_translations'] and not args.force:
            print(f"\n✅ all_translations.csv already exists, skipping translation analysis")
        else:
            print("\nLoading translations...")
            es_df = load_translation_spanish(G, paths['SPANISH_DESCRIPTION_PATH'], paths['SPANISH_REFSET_PATH'])
            fr_df = load_translation_french(G, paths['FRENCH_TRAD_VAL_PATH'])
            
            translations = {
                "French": fr_df,
                "Spanish": es_df
            }
            
            # Calculate tiers
            print("\nCalculating concept tiers...")
            
            # Context tiers
            context_tiers = {}
            for lang in translations.keys():
                t0, t1, t2 = calc_context_tiers(lang, translations, G)
                context_tiers[lang] = {'tier0': t0, 'tier1': t1, 'tier2': t2}
            
            # Depth tiers
            depth_tiers = calc_depth_tiers(G)
            
            # Similarity tiers
            similarity_tiers = calc_similarity_tiers(translations, G)
            
            # Generate comprehensive DataFrame
            print("\nGenerating comprehensive concept analysis...")
            df = generate_all_concepts_df(G, translations, context_tiers, similarity_tiers, depth_tiers)
            
            # Save all_translations.csv
            print(f"Saving all_translations.csv to {paths['ALL_TRANSLATIONS_PATH']}")
            df.to_csv(paths['ALL_TRANSLATIONS_PATH'], index=False)
            
            # Generate and save sample DataFrame
            sample_df = create_sample_df(df)
            print(f"Saving samples.csv to {paths['SAMPLE_PATH']}")
            sample_df.to_csv(paths['SAMPLE_PATH'], index=False)
            
            # Print summary statistics
            print("\nSummary Statistics:")
            print(f"Total concepts analyzed: {len(df) // len(translations):,}")
            print(f"French translations: {len(translations['French']):,}")
            print(f"Spanish translations: {len(translations['Spanish']):,}")
            print(f"Sample concepts for experimentation: {len(sample_df):,}")
    
    # Generate embeddings if needed
    if need_embeddings:
        if 'df' not in locals():        # on ne l’a pas encore
            print("\nLoading all_translations.csv for embedding generation…")
            df = pd.read_csv(paths["ALL_TRANSLATIONS_PATH"])
            print(f"  → {len(df):,} rows loaded")

        # Fixe le répertoire global utilisé par les fonctions build_*_embeddings_json
        global VEC_DIR
        VEC_DIR = str(paths["EMBEDDINGS_DIR"])  # <— harmonisation des chemins

        print("\n" + "=" * 80)
        print("GENERATING EMBEDDINGS")
        print("=" * 80)

        # Vérifie la présence de chaque fichier .pkl.gz
        bow_exists   = paths["EMBEDDINGS_DIR"] / "bow_binary_ngram.pkl.gz"
        tfidf_exists = paths["EMBEDDINGS_DIR"] / "tfidf_ngram.pkl.gz"
        st_exists    = paths["EMBEDDINGS_DIR"] / "st_multilingual.pkl.gz"

        # ----- BoW --------------------------------------------------------
        if not bow_exists.exists() or args.force:
            print("\n⚙️  Building BoW embeddings …")
            build_bow_embeddings_json(df, fname="bow_binary_ngram")
        else:
            print("✅ BoW embeddings already exist, skipping…")

        # ----- TF-IDF -----------------------------------------------------
        if not tfidf_exists.exists() or args.force:
            print("\n⚙️  Building TF-IDF embeddings …")
            build_tfidf_embeddings_json(df, fname="tfidf_ngram")
        else:
            print("✅ TF-IDF embeddings already exist, skipping…")

        # ----- Sentence-Transformer --------------------------------------
        if not st_exists.exists() or args.force:
            print("\n⚙️  Building Sentence-Transformer embeddings …")
            build_encoder_embeddings_json(df, fname="st_multilingual")
        else:
            print("✅ Sentence-Transformer embeddings already exist, skipping…")

        print("\n✅ All embeddings processed successfully!")
    else:
        print("\nSkipped embedding generation (use without --skip-embeddings to generate)")

if __name__ == "__main__":
    main()

