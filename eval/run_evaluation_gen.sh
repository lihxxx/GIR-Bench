DATASET_DIR="../dataset/reference"
MODELS_DIR="../dataset/generation/t2i"
OUTPUT_DIR="../evaluation/t2i"

COUNTING_JSON="../dataset/reference/prompt/NumericalReasoning.json"
SPATIAL_JSON="../dataset/reference/prompt/SpatialLayout.json"
TEXT_JSON="../dataset/reference/prompt/TextRendering.json"
MODEL_PATH="../weights/InternVL3_5-38B-HF"
DINOV3_MODEL_PATH=../weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
DINOV3_REPO_DIR=../dinov3

# MODELS=("YourModel1" "YourModel2")


export CUDA_VISIBLE_DEVICES=0,1

python3 eval_SpatialLayout.py \
--dataset_dir $DATASET_DIR \
--models_dir $MODELS_DIR \
--output_dir $OUTPUT_DIR \
--spatial_json $SPATIAL_JSON \
--internvl_model_path $MODEL_PATH \
# --models "${MODELS[@]}"

python3 eval_NumericalReasoning.py \
--dataset_dir $DATASET_DIR \
--models_dir $MODELS_DIR \
--output_dir $OUTPUT_DIR \
--counting_json $COUNTING_JSON \
--internvl_model_path $MODEL_PATH \
# --models "${MODELS[@]}"

python3 eval_TextRendering.py \
--dataset_dir $DATASET_DIR \
--models_dir $MODELS_DIR \
--output_dir $OUTPUT_DIR \
--text_json $TEXT_JSON \
--device cuda \
# --models "${MODELS[@]}"

python3 eval_Similarity.py \
--dataset_dir $DATASET_DIR \
--models_dir $MODELS_DIR \
--output_dir $OUTPUT_DIR \
--device auto \
--dinov3_model_path $DINOV3_MODEL_PATH \
--dinov3_repo_dir $DINOV3_REPO_DIR \
# --models "${MODELS[@]}"
