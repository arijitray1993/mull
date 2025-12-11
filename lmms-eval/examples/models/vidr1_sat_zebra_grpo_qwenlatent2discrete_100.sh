# Run as: sh examples/models/vidr1_sat_zebra_grpo_qwenlatent2discrete_100.sh
# Configuration
export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline" 

# --- Checkpoint Setup ---
# PARTIAL_GCS_PATH="vidr1_sat_zebra_grpo_mmlatent2_qwenlatent1_new/2025-10-24_03-16-49/checkpoint-132"
# PARTIAL_GCS_PATH="vidr1_sat_zebra_grpo_mmlatent2_qwenlatent1_new10/2025-10-24_22-57-48/checkpoint-100"

PARTIAL_GCS_PATH="vidr1_sat_zebra_grpo_mmlatent2discrete_qwenlatent1_100/2025-11-05_18-50-47/checkpoint-500"

# PARTIAL_GCS_PATH="vidr1_sat_zebra_grpo_mmlatent2_qwenlatent1_new_textonly/2025-10-28_01-29-09/checkpoint-400"
# PARTIAL_GCS_PATH="vidr1_sat_zebra_grpo_mmlatent2_qwenlatent1_new200/2025-10-25_19-10-26/checkpoint-350"
LOCAL_CHECKPOINTS_DIR="/home/jupyter/checkpoints"
COPY_SCRIPT="../google_scripts/google_prep_scripts/grab_checkpoints_bash.py" # Assuming the python script is in the same directory

# The full local path where the checkpoint is expected
FULL_LOCAL_PATH="${LOCAL_CHECKPOINTS_DIR}/${PARTIAL_GCS_PATH}"

# --- Conditional Checkpoint Copy ---
if [ ! -d "${FULL_LOCAL_PATH}" ]; then
    echo "Checkpoint not found locally at: ${FULL_LOCAL_PATH}"
    echo "Attempting to copy from GCS using ${COPY_SCRIPT}..."

    # Check if the copy script exists
    if [ ! -f "${COPY_SCRIPT}" ]; then
        echo "ERROR: Checkpoint copy script not found at ${COPY_SCRIPT}. Please ensure it is present."
        exit 1
    fi

    # Execute the Python copy script with the partial path
    # If the python script fails (e.g., gsutil fails or path is wrong), the '|| exit 1' will stop the script.
    python3 "${COPY_SCRIPT}" "${PARTIAL_GCS_PATH}"

    # Check the exit status of the python script
    if [ $? -ne 0 ]; then
        echo "ERROR: Checkpoint copy failed. Aborting evaluation."
        exit 1
    fi

    echo "Checkpoint successfully copied."
else
    echo "Checkpoint found locally at: ${FULL_LOCAL_PATH}"
fi

# --- LMMS-Eval Execution ---

# Define the model arguments, using the FULL_LOCAL_PATH
MODEL_ARGS="pretrained=${FULL_LOCAL_PATH},max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False"

echo "Starting LMMS Evaluation..."

accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatentdiscrete \
    --model_args="${MODEL_ARGS}" \
    --gen_kwargs=prompt_mode=mmlatent2,prompt_version=new,num_latents=100 \
    --tasks blink_iqtest,blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,sat_real,erqa,stare,vsibench \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=vidr1_sat_zebra_grpo_qwenlatent2,name=checkpoint-900


# blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,sat_real,erqa,stare,vsibench 