#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_five_shot.py – Generate five-shot translations for all retrieval methods
================================================================================

This script generates five-shot translation files for comparing different example
retrieval methods. It selects concepts that can be translated using multiple methods
to enable fair comparisons between approaches.

The script performs the following steps:
1. Load required data files (graph, translations, example retrieval results)
2. Analyze available examples for each retrieval method
3. Select concepts with maximum overlap across methods
4. Generate five-shot prompts and translations using Aya-101
5. Output TSV files for each method

Methods supported:
- bow: Bag of words similarity
- graph: Node2Vec graph embeddings  
- tfidf: TF-IDF similarity
- enc: Sentence-transformer embeddings
- random: Random examples
- rgraph: Graph-based relationships (parents, siblings, attributes)
- default: Hierarchical examples

Author: L. Megret – Inria SED (2025)
"""

from __future__ import annotations

import os
import ast
import json
import random
import time
import argparse
from pathlib import Path
from typing import Dict, Set, List, Any, Tuple
from collections import defaultdict, Counter

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

# Graph processing
from snomed_graph.snomed_graph import SnomedGraph

# ----------------------------------------------------------------------------
# Arguments and paths setup
# ----------------------------------------------------------------------------

def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Generate five-shot translations for all retrieval methods")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--max-new", type=int, default=64, help="Maximum new tokens to generate")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 20 concepts per method")
    parser.add_argument("--min-methods", type=int, default=4, help="Minimum number of methods a concept must support")
    parser.add_argument("--max-concepts", type=int, default=100, help="Maximum concepts to process per method")
    return parser.parse_args()

def setup_paths():
    """Setup all file paths for the project."""
    path_directory = os.getcwd()
    
    # Input paths
    SNOMED_GRAPH_RELATIVE_PATH = os.path.join("snomed_graph", "full_concept_graph_snomed_ct_int_rf2_20241201.gml")
    ALL_TRANSLATIONS_RELATIVE_PATH = os.path.join("data", "all_translations.csv")
    EXAMPLE_RETRIEVAL_RELATIVE_PATH = os.path.join("data", "example_retrieval_es_fr.csv")
    DEFAULT_EXAMPLE_FR_ES_RELATIVE_PATH = os.path.join("data", "default_example_fr_es.csv")
    
    # Output directory
    OUTPUT_DIR = Path(path_directory) / "data" / "five_shot_generations"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    paths = {
        'SNOMED_GRAPH_PATH': os.path.join(path_directory, SNOMED_GRAPH_RELATIVE_PATH),
        'ALL_TRANSLATIONS_PATH': os.path.join(path_directory, ALL_TRANSLATIONS_RELATIVE_PATH),
        'EXAMPLE_RETRIEVAL_PATH': os.path.join(path_directory, EXAMPLE_RETRIEVAL_RELATIVE_PATH),
        'DEFAULT_EXAMPLE_FR_ES_PATH': os.path.join(path_directory, DEFAULT_EXAMPLE_FR_ES_RELATIVE_PATH),
        'OUTPUT_DIR': OUTPUT_DIR
    }
    
    return paths

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def parse_list_field(x):
    """Parse list field that might be string representation of list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return ast.literal_eval(x)
        except Exception:
            pass
    return []

def fix_spanish_translation(ref):
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
    import re
    ref_str = re.sub(r'\s*\[como un todo\]', '', ref_str)
    return ref_str.strip()

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

# ----------------------------------------------------------------------------
# Data loading and preprocessing
# ----------------------------------------------------------------------------

def load_and_preprocess_data(paths):
    """Load and preprocess all required data files."""
    print("Loading and preprocessing data files...")
    
    # Load SNOMED CT graph
    print("  Loading SNOMED CT graph...")
    G = SnomedGraph.from_serialized(paths['SNOMED_GRAPH_PATH'])
    print(f"  Loaded graph with {len(G):,} concepts")
    
    # Load all translations
    print("  Loading all translations...")
    all_translations_df = pd.read_csv(paths['ALL_TRANSLATIONS_PATH'])
    print(f"  Loaded {len(all_translations_df):,} translation records")
    
    # Load example retrieval results
    print("  Loading example retrieval results...")
    example_retrieval_df = pd.read_csv(paths['EXAMPLE_RETRIEVAL_PATH'], sep='\t')
    print(f"  Loaded {len(example_retrieval_df):,} example retrieval records")
    
    # Load default examples
    print("  Loading default examples...")
    default_examples_df = pd.read_csv(paths['DEFAULT_EXAMPLE_FR_ES_PATH'], sep='\t')
    print(f"  Loaded {len(default_examples_df):,} default examples")
    
    # Clean Spanish translations
    print("  Cleaning Spanish translations...")
    spanish_mask = all_translations_df['language'] == 'Spanish'
    all_translations_df.loc[spanish_mask, 'reference_translations'] = \
        all_translations_df.loc[spanish_mask, 'reference_translations'].apply(fix_spanish_translation)
    
    # Filter to concepts with translations
    translations_with_trans = all_translations_df[all_translations_df['has_translation'] == True].copy()
    print(f"  Found {len(translations_with_trans):,} concepts with translations")
    
    return G, all_translations_df, translations_with_trans, example_retrieval_df, default_examples_df

# ----------------------------------------------------------------------------
# Example analysis and selection
# ----------------------------------------------------------------------------

def analyze_method_availability(example_retrieval_df, default_examples_df, translations_with_trans):
    """Analyze which concepts have sufficient examples for each method."""
    print("Analyzing method availability...")
    
    # Method availability tracking
    method_availability = defaultdict(lambda: defaultdict(set))  # method -> language -> set of sctids
    
    # Similarity-based methods (bow, tfidf, enc, graph)
    similarity_methods = ['bow', 'tfidf', 'enc', 'graph']
    
    for method in similarity_methods:
        print(f"  Analyzing {method} method...")
        method_examples = example_retrieval_df[
            (example_retrieval_df['retrieval_method'] == 'similarity') &
            (example_retrieval_df['similarity_type'] == method)
        ]
        
        # Group by source concept and language, count examples
        example_counts = method_examples.groupby(['source_sctid', 'source_language']).size()
        
        # Keep concepts with >= 5 examples
        sufficient_examples = example_counts[example_counts >= 5]
        
        for (sctid, language), count in sufficient_examples.items():
            method_availability[method][language].add(sctid)
        
        print(f"    {method}: {len(sufficient_examples)} concept-language pairs with ≥5 examples")
    
    # Graph-based relationships method (rgraph)
    print("  Analyzing rgraph method...")
    graph_examples = example_retrieval_df[example_retrieval_df['retrieval_method'] == 'graph']
    graph_counts = graph_examples.groupby(['source_sctid', 'source_language']).size()
    sufficient_graph = graph_counts[graph_counts >= 5]
    
    for (sctid, language), count in sufficient_graph.items():
        method_availability['rgraph'][language].add(sctid)
    
    print(f"    rgraph: {len(sufficient_graph)} concept-language pairs with ≥5 examples")
    
    # Default method
    print("  Analyzing default method...")
    hierarchy_counts = default_examples_df.dropna(subset=['pt_fr', 'pt_es']).groupby('hierarchy').size()
    sufficient_hierarchies = set(hierarchy_counts[hierarchy_counts >= 5].index)
    
    for language in ['French', 'Spanish']:
        concepts_in_sufficient_hierarchies = translations_with_trans[
            (translations_with_trans['language'] == language) &
            (translations_with_trans['hierarchy'].isin(sufficient_hierarchies))
        ]
        for sctid in concepts_in_sufficient_hierarchies['sctid']:
            method_availability['default'][language].add(sctid)
    
    print(f"    default: {len(sufficient_hierarchies)} hierarchies with ≥5 examples")
    
    # Random method (can use any concept with translation)
    print("  Analyzing random method...")
    for language in ['French', 'Spanish']:
        available_for_random = set(translations_with_trans[
            translations_with_trans['language'] == language
        ]['sctid'])
        
        # Random can use any concept as long as there are enough other concepts for examples
        if len(available_for_random) >= 6:  # Need 5 examples + 1 target
            method_availability['random'][language] = available_for_random
    
    print(f"    random: Available for all concepts with translations")
    
    return method_availability

def select_optimal_concepts(method_availability, min_methods=4, max_concepts=100):
    """Select concepts that can be translated by the maximum number of methods."""
    print(f"Selecting optimal concepts (min_methods={min_methods}, max_concepts={max_concepts})...")
    
    # Count methods available for each concept-language pair
    concept_method_counts = defaultdict(lambda: defaultdict(int))
    
    for method, lang_data in method_availability.items():
        for language, sctids in lang_data.items():
            for sctid in sctids:
                concept_method_counts[(sctid, language)][method] = 1
    
    # Convert to list with counts
    concept_scores = []
    for (sctid, language), methods in concept_method_counts.items():
        method_count = sum(methods.values())
        if method_count >= min_methods:
            concept_scores.append({
                'sctid': sctid,
                'language': language,
                'method_count': method_count,
                'available_methods': list(methods.keys())
            })
    
    # Sort by method count (descending) and limit
    concept_scores.sort(key=lambda x: x['method_count'], reverse=True)
    
    if max_concepts and len(concept_scores) > max_concepts:
        concept_scores = concept_scores[:max_concepts]
    
    print(f"  Selected {len(concept_scores)} concepts")
    print(f"  Method count distribution:")
    
    method_count_dist = Counter(item['method_count'] for item in concept_scores)
    for count in sorted(method_count_dist.keys(), reverse=True):
        print(f"    {count} methods: {method_count_dist[count]} concepts")
    
    return concept_scores

# ----------------------------------------------------------------------------
# Prompt generation functions
# ----------------------------------------------------------------------------

def generate_similarity_prompt(concept_info, method, example_retrieval_df, translations_with_trans):
    """Generate prompt for similarity-based methods."""
    sctid = concept_info['sctid']
    language = concept_info['language']
    
    # Get examples for this method
    method_examples = example_retrieval_df[
        (example_retrieval_df['source_sctid'] == sctid) &
        (example_retrieval_df['source_language'] == language) &
        (example_retrieval_df['retrieval_method'] == 'similarity') &
        (example_retrieval_df['similarity_type'] == method)
    ].sort_values('score', ascending=False).head(5)
    
    if len(method_examples) < 5:
        return None
    
    # Get source concept info
    source_info = translations_with_trans[
        (translations_with_trans['sctid'] == sctid) &
        (translations_with_trans['language'] == language)
    ].iloc[0]
    
    # Build prompt
    example_lines = []
    for _, example in method_examples.iterrows():
        example_lines.append(
            f'Translate the following clinical concept into {language}: "{example["example_preferred_term"]}". '
            f'{str(example["example_translation"]).strip().strip(".")}.'
        )
    
    target_line = f'Translate the following clinical concept into {language}: "{source_info["fsn"].split("(")[0].strip()}".'
    
    return "\n".join(example_lines + [target_line])

def generate_graph_prompt(concept_info, example_retrieval_df, translations_with_trans):
    """Generate prompt for graph-based method."""
    sctid = concept_info['sctid']
    language = concept_info['language']
    
    # Get graph examples for this concept
    graph_examples = example_retrieval_df[
        (example_retrieval_df['source_sctid'] == sctid) &
        (example_retrieval_df['source_language'] == language) &
        (example_retrieval_df['retrieval_method'] == 'graph')
    ].copy()
    
    if len(graph_examples) < 5:
        return None
    
    # Rank examples by relation type priority
    relation_priority = {"parent": 0, "sibling": 1, "attribute": 2, "ancestor": 3}
    
    def rank_key(row):
        rel = row.get("relation_type", "zz")
        if rel == "ancestor":
            return (relation_priority.get(rel, 99), row.get("degree_of_parenthood", 1e6))
        return (relation_priority.get(rel, 99), 0)
    
    # Sort and select top 5
    graph_examples['rank'] = graph_examples.apply(rank_key, axis=1)
    graph_examples = graph_examples.sort_values('rank').head(5)
    
    # Get source concept info
    source_info = translations_with_trans[
        (translations_with_trans['sctid'] == sctid) &
        (translations_with_trans['language'] == language)
    ].iloc[0]
    
    # Build prompt
    example_lines = []
    for _, example in graph_examples.iterrows():
        example_lines.append(
            f'Translate the following clinical concept into {language}: "{example["example_preferred_term"]}". '
            f'{str(example["example_translation"]).strip().strip(".")}.'
        )
    
    target_line = f'Translate the following clinical concept into {language}: "{source_info["fsn"].split("(")[0].strip()}".'
    
    return "\n".join(example_lines + [target_line])

def generate_default_prompt(concept_info, default_examples_df, translations_with_trans):
    """Generate prompt for default hierarchical method."""
    sctid = concept_info['sctid']
    language = concept_info['language']
    
    # Get source concept info
    source_info = translations_with_trans[
        (translations_with_trans['sctid'] == sctid) &
        (translations_with_trans['language'] == language)
    ].iloc[0]
    
    hierarchy = source_info['hierarchy']
    
    # Get examples from same hierarchy
    hierarchy_examples = default_examples_df[
        default_examples_df['hierarchy'] == hierarchy
    ].dropna(subset=['pt_fr', 'pt_es'])
    
    if len(hierarchy_examples) < 5:
        return None
    
    # Sample 5 examples
    examples = hierarchy_examples.sample(5, replace=False)
    
    # Build prompt
    lang_col = 'pt_fr' if language == 'French' else 'pt_es'
    example_lines = []
    
    for _, example in examples.iterrows():
        example_lines.append(
            f'Translate the following clinical concept into {language}: "{example["pt_en"]}". '
            f'{str(example[lang_col]).strip().strip(".")}.'
        )
    
    target_line = f'Translate the following clinical concept into {language}: "{source_info["fsn"].split("(")[0].strip()}".'
    
    return "\n".join(example_lines + [target_line])

def generate_random_prompt(concept_info, translations_with_trans):
    """Generate prompt for random method."""
    sctid = concept_info['sctid']
    language = concept_info['language']
    
    # Get source concept info
    source_info = translations_with_trans[
        (translations_with_trans['sctid'] == sctid) &
        (translations_with_trans['language'] == language)
    ].iloc[0]
    
    # Get random examples (excluding the source concept)
    available_examples = translations_with_trans[
        (translations_with_trans['language'] == language) &
        (translations_with_trans['sctid'] != sctid)
    ]
    
    if len(available_examples) < 5:
        return None
    
    # Sample 5 random examples
    examples = available_examples.sample(5, replace=False)
    
    # Build prompt
    example_lines = []
    for _, example in examples.iterrows():
        pt_en = example['fsn'].split('(')[0].strip()
        translation = str(example['reference_translations']).strip().strip('.')
        example_lines.append(
            f'Translate the following clinical concept into {language}: "{pt_en}". {translation}.'
        )
    
    target_line = f'Translate the following clinical concept into {language}: "{source_info["fsn"].split("(")[0].strip()}".'
    
    return "\n".join(example_lines + [target_line])

# ----------------------------------------------------------------------------
# Model setup and generation
# ----------------------------------------------------------------------------

def setup_model():
    """Setup Aya-101 model for translation generation."""
    print("Setting up Aya-101 model...")
    
    model_name = "CohereForAI/aya-101"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def load_model(cpu=False):
        if cpu:
            return AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map={"": "cpu"},
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_threshold=6.0,
        )
        
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    
    try:
        model = load_model(cpu=False)
    except (ValueError, RuntimeError):
        print("  VRAM insufficient, falling back to CPU...")
        model = load_model(cpu=True)
    
    device = next(model.parameters()).device
    print(f"  Model loaded on {device}")
    
    return model, tokenizer, device

@torch.inference_mode()
def generate_translations(prompts, model, tokenizer, device, batch_size=16, max_new_tokens=64):
    """Generate translations using the model."""
    results = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating translations"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Decode
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend([result.strip() for result in batch_results])
    
    return results

# ----------------------------------------------------------------------------
# Main processing function
# ----------------------------------------------------------------------------

def process_method(method, selected_concepts, data_dict, model, tokenizer, device, args):
    """Process a single method and generate translations."""
    print(f"\nProcessing {method} method...")
    
    # Filter concepts that support this method
    method_concepts = [c for c in selected_concepts if method in c['available_methods']]
    
    if args.test:
        method_concepts = method_concepts[:20]
    
    if not method_concepts:
        print(f"  No concepts available for {method} method")
        return None
    
    print(f"  Processing {len(method_concepts)} concepts")
    
    # Generate prompts
    prompts = []
    concept_data = []
    
    for concept_info in tqdm(method_concepts, desc=f"Generating {method} prompts"):
        if method in ['bow', 'tfidf', 'enc', 'graph']:
            prompt = generate_similarity_prompt(
                concept_info, method, data_dict['example_retrieval_df'], data_dict['translations_with_trans']
            )
        elif method == 'rgraph':
            prompt = generate_graph_prompt(
                concept_info, data_dict['example_retrieval_df'], data_dict['translations_with_trans']
            )
        elif method == 'default':
            prompt = generate_default_prompt(
                concept_info, data_dict['default_examples_df'], data_dict['translations_with_trans']
            )
        elif method == 'random':
            prompt = generate_random_prompt(
                concept_info, data_dict['translations_with_trans']
            )
        else:
            continue
        
        if prompt:
            prompts.append(prompt)
            
            # Get concept details
            source_info = data_dict['translations_with_trans'][
                (data_dict['translations_with_trans']['sctid'] == concept_info['sctid']) &
                (data_dict['translations_with_trans']['language'] == concept_info['language'])
            ].iloc[0]
            
            concept_data.append({
                'sctid': concept_info['sctid'],
                'preferred_term': source_info['fsn'].split('(')[0].strip(),
                'language': concept_info['language'],
                'ref_traduction': source_info['reference_translations'],
                'hierarchy': source_info['hierarchy'],
                f'prompt_{method}_five_shot': prompt
            })
    
    if not prompts:
        print(f"  No valid prompts generated for {method}")
        return None
    
    print(f"  Generated {len(prompts)} valid prompts")
    
    # Generate translations
    translations = generate_translations(
        prompts, model, tokenizer, device, 
        batch_size=args.batch_size, max_new_tokens=args.max_new
    )
    
    # Add translations to concept data
    for i, translation in enumerate(translations):
        concept_data[i]['aya_translation'] = translation
    
    # Create DataFrame
    result_df = pd.DataFrame(concept_data)
    
    return result_df

def main():
    """Main execution function."""
    print("=" * 80)
    print("SNOMED CT Five-Shot Translation Generation")
    print("=" * 80)
    
    # Setup
    args = setup_arguments()
    paths = setup_paths()
    
    print(f"Configuration:")
    print(f"  Test mode: {args.test}")
    print(f"  Min methods per concept: {args.min_methods}")
    print(f"  Max concepts: {args.max_concepts}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max new tokens: {args.max_new}")
    
    # Load data
    G, all_translations_df, translations_with_trans, example_retrieval_df, default_examples_df = \
        load_and_preprocess_data(paths)
    
    data_dict = {
        'G': G,
        'all_translations_df': all_translations_df,
        'translations_with_trans': translations_with_trans,
        'example_retrieval_df': example_retrieval_df,
        'default_examples_df': default_examples_df
    }
    
    # Analyze method availability
    method_availability = analyze_method_availability(
        example_retrieval_df, default_examples_df, translations_with_trans
    )
    
    # Select optimal concepts
    selected_concepts = select_optimal_concepts(
        method_availability, args.min_methods, args.max_concepts
    )
    
    if not selected_concepts:
        print("No concepts selected. Exiting.")
        return
    
    # Setup model
    model, tokenizer, device = setup_model()
    
    # Process each method
    methods = ['bow', 'graph', 'tfidf', 'enc', 'random', 'rgraph', 'default']
    
    for method in methods:
        print(f"\n{'='*60}")
        result_df = process_method(method, selected_concepts, data_dict, model, tokenizer, device, args)
        
        if result_df is not None:
            # Save result
            suffix = "_test" if args.test else ""
            output_file = paths['OUTPUT_DIR'] / f"five_shot_{method}{suffix}.tsv"
            result_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
            print(f"  ✅ Saved {len(result_df)} translations to {output_file}")
        else:
            print(f"  ❌ No results for {method} method")
    
    print(f"\n{'='*80}")
    print("Five-shot generation completed!")
    print(f"Results saved in: {paths['OUTPUT_DIR']}")
    
    # Print summary
    generated_files = list(paths['OUTPUT_DIR'].glob("five_shot_*.tsv"))
    print(f"\nGenerated files ({len(generated_files)}):")
    for file_path in sorted(generated_files):
        df = pd.read_csv(file_path, sep='\t')
        print(f"  {file_path.name}: {len(df)} translations")

if __name__ == "__main__":
    main()
