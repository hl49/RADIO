#!/bin/bash

#SBATCH -p alldlc_gpu-rtx2080        # Specify the partition (queue) to use
#SBATCH --exclude=dlcgpu08,dlcgpu17           # Exclude this problematic node
#SBATCH --job-name=radio-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 # better to 6
#SBATCH --mem=16G   # better to 24G
#SBATCH --mail-type=END,FAIL         # (recive mails about end and timeouts/crashes of your job)
#SBATCH --mail-user=henrylema94@gmail.com  # Specify your email address


# Function to check GPU health
check_gpu_health() {
    echo "=== GPU Health Check ==="
    
    # Check for GPU availability
    gpu_check=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    echo "Number of GPUs detected: $gpu_check"
    
    if [ "$gpu_check" -eq 0 ]; then
        echo "No GPU available. Exiting."
        exit 1
    fi
    
    # Run nvidia-smi and check for ERR! messages
    echo "Running nvidia-smi check..."
    smi_output=$(nvidia-smi)
    echo "$smi_output"
    
    if echo "$smi_output" | grep -q "ERR!"; then
        echo "[ERR] GPU error detected in nvidia-smi output. Exiting."
        exit 1
    fi
    
    echo "GPU health check passed!"
    echo "=== GPU Health Check Complete ==="
    echo ""
}


# Move to project directory
cd /work/dlclarge2/lemah-thesis/RADIO/mmseg

# Set dataset root
export VOC_ROOT_DIR=/work/dlclarge2/lemah-thesis/RADIO/mmseg/data/VOCdevkit/VOC2012/


# Running the job
start=$(date +%s)

# Run GPU health check before training
check_gpu_health

# Train using the Singularity container
singularity exec --nv radio_env.sif \
  python -m torch.distributed.launch \
  --nnodes=1 \
  --nproc_per_node=2 \
  train.py configs/radio/radio_b_linear_8xb2-80k_voc-512x512.py \
  --launcher pytorch \
  --cfg-options "train_dataloader.dataset.data_root=/${VOC_ROOT_DIR}" \
  --cfg-options "val_dataloader.dataset.data_root=${VOC_ROOT_DIR}"


end=$(date +%s)
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: ${runtime} seconds"