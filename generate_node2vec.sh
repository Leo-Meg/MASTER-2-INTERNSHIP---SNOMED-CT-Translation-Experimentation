#!/bin/bash
#SBATCH --job-name=node2vec_embeddings
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.megret@inria.fr

# Configuration environnement
source /home/${USER}/.bashrc

# Initialize conda and activate environment
eval "$(conda shell.bash hook)"
conda activate env_node2vec

# Répertoire de travail
cd /home/${USER}/scratch/NRC_test

echo "Starting Node2Vec embeddings generation..."
echo "Allocated memory: ${SLURM_MEM_PER_NODE}MB"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

# Vérifications
if [[ ! -f "generate_node2vec_embeddings.py" ]]; then
    echo "Error: generate_node2vec_embeddings.py not found"
    exit 1
fi

if [[ ! -f "full_concept_graph_snomed_ct_int_rf2_20241201.gml" ]]; then
    echo "Error: SNOMED graph not found"
    exit 1
fi

# Créer les dossiers de sortie
mkdir -p data/embeddings

# Monitoring mémoire en arrière-plan
(while true; do
    echo "Memory usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    sleep 300  # Toutes les 5 minutes
done) &
MONITOR_PID=$!

# Exécuter le script Python avec échantillonnage
echo "Running Node2Vec generation with sampling..."
python generate_node2vec_sampled.py --max-nodes 30000

# Arrêter le monitoring
kill $MONITOR_PID 2>/dev/null || true

echo "✅ Node2Vec embeddings generation completed!"
echo "Generated files:"
ls -lh data/embeddings/
