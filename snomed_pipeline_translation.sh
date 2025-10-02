#!/bin/bash
#SBATCH --job-name=snomed_pipeline
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.megret@inria.fr

##############################################################################
#  Pipeline SNOMED CT Translations
#  ===============================
#  
#  Ce script exécute dans l'ordre :
#  1. load_translations_embedding.py  - Chargement traductions + génération embeddings (sans Node2Vec)
#  2. retrieval_example.py            - Génération exemples de récupération
#  3. generate_five_shot.py           - Génération traductions five-shot
#
#  IMPORTANT: Les embeddings Node2Vec doivent être générés séparément avec :
#             sbatch generate_node2vec.sh
#
#  Utilisation :
#  ------------
#  Exécution complète :
#      sbatch pipeline_snomed_translations.sh
#
#  Exécution test (traitement limité) :
#      sbatch pipeline_snomed_translations.sh test
#
#  Exécution sans embeddings (si déjà générés) :
#      sbatch pipeline_snomed_translations.sh skip-embeddings
##############################################################################

# ----------------------------------------------------------------------
# 1) Détection des modes d'exécution
# ----------------------------------------------------------------------
TEST_MODE=false
SKIP_EMBEDDINGS=false
EXTRA_ARGS=""

for arg in "$@"; do
  case $arg in
    test)
      TEST_MODE=true
      EXTRA_ARGS="$EXTRA_ARGS --test"
      echo "[PIPELINE] Mode test activé"
      ;;
    skip-embeddings)
      SKIP_EMBEDDINGS=true
      echo "[PIPELINE] Génération d'embeddings désactivée"
      ;;
  esac
done

# ----------------------------------------------------------------------
# 2) Configuration environnement
# ----------------------------------------------------------------------
source /home/${USER}/.bashrc

# Initialize conda and activate environment
echo "Activating conda environment: env_snomed_trans_poc"
eval "$(conda shell.bash hook)"
conda activate env_snomed_trans_poc

# Configuration parallélisme
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ----------------------------------------------------------------------
# 3) Vérification des dépendances (SANS Node2Vec)
# ----------------------------------------------------------------------
echo "[PIPELINE] Installation/vérification des dépendances..."
pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    tqdm \
    sentence-transformers \
    networkx \
    faiss-cpu \
    joblib \
    torch \
    transformers \
    accelerate \
    bitsandbytes \
    git+https://github.com/VerataiLtd/snomed_graph.git@main#egg=snomed_graph

# ----------------------------------------------------------------------
# 4) Répertoire de travail
# ----------------------------------------------------------------------
cd /home/${USER}/scratch/NRC_test

# Vérification présence des fichiers requis
echo "[PIPELINE] Vérification des fichiers requis..."
REQUIRED_FILES=(
    "load_translations_embedding.py"
    "retrieval_example.py" 
    "generate_five_shot.py"
    "snomed_graph/full_concept_graph_snomed_ct_int_rf2_20241201.gml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "[PIPELINE] ERREUR: Fichier manquant: $file"
        exit 1
    fi
done

# Vérification présence des répertoires SNOMED
if [[ ! -d "SnomedCT_FR" ]] || [[ ! -d "SnomedCT_SpanishRelease-es_PRODUCTION_20240930T120000Z" ]]; then
    echo "[PIPELINE] ERREUR: Répertoires SNOMED manquants"
    exit 1
fi

echo "[PIPELINE] Tous les fichiers requis sont présents"

# ----------------------------------------------------------------------
# 5) Exécution Étape 1: Chargement traductions + génération embeddings
# ----------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "ÉTAPE 1: Chargement traductions et génération embeddings"
echo "=================================================================="

if [[ "$SKIP_EMBEDDINGS" == true ]]; then
    echo "[PIPELINE] Génération d'embeddings sautée (skip-embeddings activé)"
    srun python load_translations_embedding.py --skip-embeddings
else
    echo "[PIPELINE] Génération complète avec embeddings"
    srun python load_translations_embedding.py
fi

# Vérification succès étape 1
if [[ $? -ne 0 ]]; then
    echo "[PIPELINE] ERREUR: Échec de l'étape 1 (load_translations_embedding.py)"
    exit 1
fi

echo "[PIPELINE] ✅ Étape 1 terminée avec succès"

# ----------------------------------------------------------------------
# 6) Exécution Étape 2: Génération exemples de récupération
# ----------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "ÉTAPE 2: Génération exemples de récupération"
echo "=================================================================="

srun python retrieval_example.py

# Vérification succès étape 2
if [[ $? -ne 0 ]]; then
    echo "[PIPELINE] ERREUR: Échec de l'étape 2 (retrieval_example.py)"
    exit 1
fi

echo "[PIPELINE] ✅ Étape 2 terminée avec succès"

# ----------------------------------------------------------------------
# 7) Exécution Étape 3: Génération traductions five-shot
# ----------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "ÉTAPE 3: Génération traductions five-shot"
echo "=================================================================="

if [[ "$TEST_MODE" == true ]]; then
    echo "[PIPELINE] Mode test activé pour five-shot"
    srun python generate_five_shot.py --test --batch-size 8 --max-concepts 50
else
    echo "[PIPELINE] Mode complet pour five-shot"
    srun python generate_five_shot.py --batch-size 16 --max-new 64
fi

# Vérification succès étape 3
if [[ $? -ne 0 ]]; then
    echo "[PIPELINE] ERREUR: Échec de l'étape 3 (generate_five_shot.py)"
    exit 1
fi

echo "[PIPELINE] ✅ Étape 3 terminée avec succès"

# ----------------------------------------------------------------------
# 8) Rapport final
# ----------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "PIPELINE TERMINÉ AVEC SUCCÈS"
echo "=================================================================="

# Statistiques des fichiers générés
echo "[PIPELINE] Fichiers générés:"
if [[ -f "data/all_translations.csv" ]]; then
    echo "  ✅ data/all_translations.csv ($(wc -l < data/all_translations.csv) lignes)"
fi

if [[ -f "data/samples.csv" ]]; then
    echo "  ✅ data/samples.csv ($(wc -l < data/samples.csv) lignes)"
fi

if [[ -f "data/example_retrieval_es_fr.csv" ]]; then
    echo "  ✅ data/example_retrieval_es_fr.csv ($(wc -l < data/example_retrieval_es_fr.csv) lignes)"
fi

if [[ -d "data/embeddings" ]]; then
    echo "  ✅ data/embeddings/ ($(ls -1 data/embeddings/ | wc -l) fichiers)"
fi

if [[ -d "data/five_shot_generations" ]]; then
    echo "  ✅ data/five_shot_generations/ ($(ls -1 data/five_shot_generations/ | wc -l) fichiers)"
fi

# Espace disque utilisé
echo "[PIPELINE] Espace disque utilisé:"
du -sh data/ 2>/dev/null || echo "  Répertoire data/ non trouvé"

echo "[PIPELINE] Pipeline terminé à $(date)"
echo "=================================================================="
