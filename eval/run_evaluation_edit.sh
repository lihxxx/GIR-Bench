DATASET_DIR="../dataset/reference"
MODELS_DIR="../dataset/generation/editing/"
OUTPUT_DIR="../evaluation/editing"

# MODELS=("YourModel1" "YourModel2")


python3 eval_ReasoningPerception.py \
    --dataset_dir $DATASET_DIR \
    --models_dir $MODELS_DIR \
    --output_dir $OUTPUT_DIR \
    # --models "${MODELS[@]}"


python3 eval_VisualLogic.py \
    --dataset_dir $DATASET_DIR \
    --models_dir $MODELS_DIR \
    --output_dir $OUTPUT_DIR \
    # --models "${MODELS[@]}"

python3 eval_VisualPuzzle.py \
    --dataset_dir $DATASET_DIR \
    --models_dir $MODELS_DIR \
    --output_dir $OUTPUT_DIR \
    # --models "${MODELS[@]}"

