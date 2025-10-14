import os
import json
import argparse
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils import (
    NumpyEncoder,
    ensure_dir,
    resolve_models,
    write_json,
    list_generated_images,
    parse_image_id,
    load_internvl_model,
    denormalize_bboxes,
    model_inference,
    build_detection_prompt,
    extract_grounding_from_response,
    process_detection_result,
)


def load_model(model_path: str,
               device: str = 'cuda') -> Tuple[Any, Any, None]:
    logging.info(f"Loading InternVL model: {model_path} ...")
    model, processor, tok = load_internvl_model(
        model_path,
        device=device,
    )
    logging.info("InternVL model loaded.")

    return model, processor, tok

def count_objects(model: Any,
                  processor: Any,
                  image_path: str,
                  object_names: List[str]) -> Tuple[Dict[str, int], Dict[str, Any], str, List[str], List[str], List[str], str]:
    image = Image.open(image_path).convert('RGB')
    object_descriptions, combined_prompt = build_detection_prompt(object_names)
    original_hw = (image.size[1], image.size[0])
    response = model_inference(model, processor, image, combined_prompt, max_new_tokens=1024)
    result_json = extract_grounding_from_response(response, object_names)
    object_counts, bbox_dict, missing_objects, excess_objects = process_detection_result(result_json, object_names)
    bbox_dict = denormalize_bboxes(bbox_dict, original_hw)

    return object_counts, bbox_dict, response, missing_objects, excess_objects, object_descriptions, combined_prompt


def evaluate_counting_image(model: Any,
                            processor: Any,
                            image_path: str,
                            task_info: Dict[str, Any]) -> Dict[str, Any]:
    object_names = task_info["objects"]
    expected_counts = task_info["number"]

    actual_counts, bbox_dict, response_text, missing_objects, excess_objects, object_descriptions, model_prompt = count_objects(
        model, processor, image_path, object_names
    )


    all_counts_correct = all(actual_counts.get(obj, 0) == count for obj, count in zip(object_names, expected_counts))
    no_missing = len(missing_objects) == 0

    result = {
        "expected": {obj: count for obj, count in zip(object_names, expected_counts)},
        "actual": actual_counts,
        "overall_correct": bool(all_counts_correct and no_missing),
        "matches": bool(all_counts_correct and no_missing),                    
        "bboxes": bbox_dict,
        "missing_objects": missing_objects,
        "excess_objects": excess_objects,
        "model_response": response_text,
        "object_descriptions": object_descriptions,
        "model_prompt": model_prompt,
    }
    return result



def process_tasks(model: Any,
                  processor: Any,
                  tasks: List[Dict[str, Any]],
                  img_dir: str,
                  output_base_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    task_results: List[Dict[str, Any]] = []
    task_correct_count = 0


    image_files = list_generated_images(model_dir:=os.path.dirname(img_dir), task_subdir=os.path.basename(img_dir))
    id_to_image_path: Dict[str, str] = {}
    for p in image_files:
        base = os.path.splitext(os.path.basename(p))[0]
        pid = parse_image_id(base) or base
        id_to_image_path[pid] = p

    for i, task in enumerate(tqdm(tasks, desc="Evaluating counting tasks")):
        task_id = task.get("id", i + 1)
        image_key = parse_image_id(str(task_id)) or f"{int(task_id):03d}" if str(task_id).isdigit() else str(task_id)
        image_path = id_to_image_path.get(image_key)

        if not image_path or not os.path.exists(image_path):
            logging.warning(f"Image not found for id={task_id} under {img_dir}, skipping")
            continue

        result = evaluate_counting_image(
            model, processor, image_path, task["task"]
        )


        if result.get('matches', False):
            task_correct_count += 1

        result["id"] = task_id
        result["prompt"] = task["prompt"]
        task_results.append(result)

    accuracy = task_correct_count / len(task_results) if task_results else 0.0
    stats: Dict[str, Any] = {
        "accuracy": accuracy,
        "total": len(task_results),
        "correct": task_correct_count,
    }


    return task_results, stats


def evaluate_model_counting(model_name: str,
                            args: argparse.Namespace) -> Dict[str, Any]:
    print(f"Evaluating model {model_name} on NumericalReasoning task ...")
    model_dir = os.path.join(args.models_dir, model_name)
    img_dir = os.path.join(model_dir, 'NumericalReasoning')
    if not os.path.exists(img_dir):
        print(f"Warning: NumericalReasoning images directory does not exist: {img_dir}")
        return {
            'error': f"Warning: NumericalReasoning images directory does not exist: {img_dir}",
        }


    eval_model, eval_processor, _ = load_model(
        args.internvl_model_path,
        device=args.device,
    )


    counting_json = args.counting_json or os.path.join(args.dataset_dir, 'benchmark', 'prompt', 'NumericalReasoning.json')
    if not os.path.exists(counting_json):
        alt = os.path.join(args.dataset_dir, 'prompt', 'NumericalReasoning.json')
        counting_json = alt if os.path.exists(alt) else counting_json
    with open(counting_json, 'r', encoding='utf-8') as f:
        tasks = json.load(f)


    model_output_dir = os.path.join(args.output_dir, model_name)
    ensure_dir(model_output_dir)


    task_results, stats = process_tasks(
        eval_model, eval_processor, tasks, img_dir, model_output_dir
    )


    model_summary = {
        'task': 'NumericalReasoning',
        'model': model_name,
        'stats': stats,
    }


    write_json(os.path.join(model_output_dir, 'numericalreasoning.json'), {"NumericalReasoning": stats, "overall_score": stats.get('accuracy', 0.0)})

    print(f"Completed model {model_name} - NumericalReasoning: {stats['correct']}/{stats['total']} correct")
    return model_summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch evaluate models on Text-to-Image NumericalReasoning task')
    parser.add_argument('--dataset_dir', default='', help='Dataset root directory')
    parser.add_argument('--models_dir', default='', help='Model outputs root directory (images)')
    parser.add_argument('--output_dir', default='', help='Evaluation outputs directory')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate (default: all subdirectories in models_dir)')
    parser.add_argument('--internvl_model_path', type=str, default='', help='InternVL model path or hub id')
    parser.add_argument('--device', type=str, default='cuda', help='Device for evaluator (cuda/cpu)')


    parser.add_argument('--counting_json', type=str, default=None, help='Path to NumericalReasoning JSON')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, d))]

    print(f"Models to evaluate: {models}")
    print("Task: NumericalReasoning")
    print("-" * 50)
    all_results: Dict[str, Any] = {}
    for model_name in models:
        model_summary = evaluate_model_counting(model_name, args)
        all_results[model_name] = model_summary

    minified: Dict[str, float] = {}
    for model_name, summary in all_results.items():
        stats = summary.get('stats', {}) if isinstance(summary, dict) else {}
        minified[model_name] = float(stats.get('accuracy', 0.0))

    summary_json = os.path.join(args.output_dir, 'NumericalReasoning_summary.json')
    write_json(summary_json, minified)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()


