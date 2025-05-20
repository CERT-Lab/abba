#!/bin/bash

# Configuration
MODEL="meta-llama/Llama-3.2-1B"
GPU_ID=0  # Specify which GPU to use

# List of run directories to process; add trained adapter directories here
RUN_DIRS=("")

# Process each run directory
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo "=== Processing: $RUN_DIR ==="
    
    # Extract method from the directory structure (one folder above run directory)
    METHOD=$(basename "$(dirname "$RUN_DIR")")
    echo "Extracted method: $METHOD"
    
    # Get the final and merged model paths
    FINAL_MODEL_PATH="$RUN_DIR/final_model"
    MERGED_MODEL_PATH="$RUN_DIR/merged_model"
    
    echo "=== Starting Merging ==="
    if [ ! -d "$FINAL_MODEL_PATH" ]; then
        echo "Error: Final model not found at $FINAL_MODEL_PATH"
        continue
    fi

    # Check if merged model already exists
    if [ -d "$MERGED_MODEL_PATH" ]; then
        echo "Merged model already exists at $MERGED_MODEL_PATH, skipping merge step"
    else
        echo "Merged model not found, proceeding with merge"
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python merge_save.py \
            --base_model "$MODEL" \
            --adapter_path "$FINAL_MODEL_PATH" \
            --output_path "$MERGED_MODEL_PATH"
    fi
    
    echo "=== Starting Evaluation ==="
    # Check if merged model exists
    if [ ! -d "$MERGED_MODEL_PATH" ]; then
        echo "Error: Merged model not found at $MERGED_MODEL_PATH"
        continue
    fi

    declare -a datasets=(
        "ARC-Challenge"
        "ARC-Easy"
        "boolq"
        "hellaswag"
        "openbookqa"
        "piqa"
        "social_i_qa"
        "winogrande"
    )

    # Loop through datasets and evaluate
    for dataset in "${datasets[@]}"; do
        echo "=== Evaluating on $dataset ==="
        
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/commonsense_eval.py \
            --model "$MERGED_MODEL_PATH" \
            --dataset "$dataset" \
            --data_file "data/commonsense/$dataset/test.json" \
            --batch_size 128 \
            --tensor_parallel_size 1 \
            --run_dir "$RUN_DIR" 
    done

    # Clean up merged model directory
    echo "=== Cleaning up merged model ==="
    if [ -d "$MERGED_MODEL_PATH" ]; then
        echo "Removing merged model directory: $MERGED_MODEL_PATH"
        rm -rf "$MERGED_MODEL_PATH"
        if [ $? -eq 0 ]; then
            echo "Successfully removed merged model directory"
        else
            echo "Warning: Failed to remove merged model directory"
        fi
    else
        echo "Merged model directory not found - nothing to clean up"
    fi
    
    echo "=== Processing Complete for $RUN_DIR ==="
    
    # Save run information
    echo "Saving run information..."
    cat << EOF > "$RUN_DIR/run_info.txt"

Run processed at: $(date)
Method: $METHOD
Base model: $MODEL
EOF

done

echo "=== All Run Directories Processed ==="