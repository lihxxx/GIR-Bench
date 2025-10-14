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
               device: str = 'cuda'):
    logging.info(f"Loading InternVL model: {model_path} ...")
    model, processor, tok = load_internvl_model(
        model_path,
        device=device,
    )
    logging.info("InternVL model loaded.")
    return model, processor, tok


def get_object_bbox(obj_name: str, bbox_dict: Dict[str, Any]):
    box = None
    normalized = obj_name.replace(' ', '_')
    if obj_name in bbox_dict:
        box = bbox_dict[obj_name]
    elif normalized in bbox_dict:
        box = bbox_dict[normalized]
    else:
        for key in bbox_dict:
            if key.startswith(f"{obj_name}_") and key.split('_')[-1].isdigit():
                box = bbox_dict[key]
                break
            if key.startswith(f"{normalized}_") and key.split('_')[-1].isdigit():
                box = bbox_dict[key]
                break
    if box is None or len(box) < 4:
        return None
    if isinstance(box, list) and len(box) > 0 and isinstance(box[0], list):
        box = box[0]
        if not box or len(box) < 4:
            return None
    return [float(c) for c in box[:4]]


def merge_group_bboxes(group_objects: List[str], bbox_dict: Dict[str, Any]):
    boxes = []
    for obj in group_objects:
        box = get_object_bbox(obj, bbox_dict)
        if box:
            boxes.append(box)
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def check_relation(subject, obj, direction: str, bbox_dict: Dict[str, Any], position_threshold: float = 0.25) -> bool:
    import numpy as np
    subj_box = None
    obj_box = None
    if isinstance(subject, list):
        subj_box = merge_group_bboxes(subject, bbox_dict)
    else:
        subj_box = get_object_bbox(subject, bbox_dict)
    if isinstance(obj, list):
        obj_box = merge_group_bboxes(obj, bbox_dict)
    else:
        obj_box = get_object_bbox(obj, bbox_dict)
    if subj_box is None or obj_box is None:
        return False
    subj_box = np.array(subj_box[:4])
    obj_box = np.array(obj_box[:4])
    boxes = np.array([
        [[subj_box[0], subj_box[1]], [subj_box[2], subj_box[3]]],
        [[obj_box[0], obj_box[1]], [obj_box[2], obj_box[3]]]
    ])
    center_a, center_b = boxes.mean(axis=1)
    dim_a = np.abs(boxes[0, 1] - boxes[0, 0])
    dim_b = np.abs(boxes[1, 1] - boxes[1, 0])
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return False
    norm = np.linalg.norm(offset)
    if norm <= 0:
        return False
    dx, dy = offset / norm
    thr = 0.5
    if direction == 'left':
        return dx < -thr
    if direction == 'right':
        return dx > thr
    if direction in ['up', 'above']:
        return dy < -thr
    if direction in ['down', 'below']:
        return dy > thr
    return False


def prepare_colored_objects(objects: List[str], colors: List[str]):
    if not colors or len(colors) == 0:
        return objects
    colored = []
    for i, obj in enumerate(objects):
        if i < len(colors) and colors[i]:
            colored.append({'name': obj, 'color': colors[i]})
        else:
            colored.append({'name': obj, 'color': ''})
    return colored


def check_spatial_relation(model, processor, image_path: str, objects: List[Any], relations: List[Dict[str, Any]], colors: List[str] = None):
    image = Image.open(image_path).convert('RGB')
    object_descriptions, detection_prompt = build_detection_prompt(objects)
    original_hw = (image.size[1], image.size[0])
    detection_response = model_inference(model, processor, image, detection_prompt, max_new_tokens=1024)

    if isinstance(objects, list) and all(isinstance(obj, dict) for obj in objects):
        object_names = [f"{obj['color']} {obj['name']}" if obj.get('color') else obj['name'] for obj in objects]
    else:
        object_names = objects

    result_json = extract_grounding_from_response(detection_response, object_names)
    object_counts, bbox_dict, missing_objects, excess_objects = process_detection_result(result_json, object_names)
    bbox_dict = denormalize_bboxes(bbox_dict, original_hw)
    if missing_objects is None:
        missing_objects = []

    results: Dict[str, Any] = {
        "all_objects_detected": False,
    }

    color_prefixes = [f"{c} " for c in (colors or []) if c]


    detected_objects_keys = set(bbox_dict.keys())

    def extract_base_name(key: str) -> str:

        if '_' in key:
            parts = key.split('_')
            if parts[-1].isdigit():
                base = '_'.join(parts[:-1])
                return base.replace('_', ' ')
        return key

    detected_base_objects = set()
    object_count_check: Dict[str, int] = {}
    for obj_key in detected_objects_keys:
        base_name = extract_base_name(obj_key)
        detected_base_objects.add(base_name)
        object_count_check[base_name] = object_count_check.get(base_name, 0) + 1

    def normalize_name(name: str) -> str:
        return name.replace(' ', '_')

    def check_object_detected(expected_obj: str) -> bool:
        if expected_obj in detected_base_objects:
            return True
        normalized = normalize_name(expected_obj)
        if normalized in detected_objects_keys:
            return True
        if any(k.startswith(f"{normalized}_") and k.split('_')[-1].isdigit() for k in detected_objects_keys):
            return True
        if any(k.startswith(f"{expected_obj}_") and k.split('_')[-1].isdigit() for k in detected_objects_keys):
            return True
        return False

    def count_detected_objects(expected_obj: str) -> int:
        count = 0
        normalized = normalize_name(expected_obj)
        for k in detected_objects_keys:
            if k == expected_obj or k == normalized:
                count += 1
            elif k.startswith(f"{normalized}_") and k.split('_')[-1].isdigit():
                count += 1
            elif k.startswith(f"{expected_obj}_") and k.split('_')[-1].isdigit():
                count += 1
        return count

    all_detected = True
    for obj in object_names:
        if not check_object_detected(obj):
            if obj not in missing_objects:
                all_detected = False
                break


    if all_detected and not missing_objects:
        for obj in object_names:
            if count_detected_objects(obj) != 1:
                all_detected = False
                break


    results["all_objects_detected"] = bool(all_detected and not missing_objects and not excess_objects)


    relationship_results: Dict[str, bool] = {}
    if isinstance(relations, list) and len(relations) > 0:
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            subject_idx = rel.get('subject')
            direction = rel.get('direction')
            object_idx = rel.get('object')
            subject = object_names[subject_idx] if isinstance(subject_idx, int) and 0 <= subject_idx < len(object_names) else subject_idx
            obj = object_names[object_idx] if isinstance(object_idx, int) and 0 <= object_idx < len(object_names) else object_idx
            subj_missing = False
            obj_missing = False
            if isinstance(subject, list):
                subj_missing = any(s in missing_objects for s in subject)
            else:
                subj_missing = subject in missing_objects
            if isinstance(obj, list):
                obj_missing = any(o in missing_objects for o in obj)
            else:
                obj_missing = obj in missing_objects
            if subj_missing or obj_missing:
                subj_str = ','.join(subject) if isinstance(subject, list) else subject
                obj_str = ','.join(obj) if isinstance(obj, list) else obj
                relationship_results[f"{subj_str} {direction} {obj_str}"] = False
                continue
            is_correct = check_relation(subject, obj, direction, bbox_dict)
            subj_str = ','.join(subject) if isinstance(subject, list) else subject
            obj_str = ','.join(obj) if isinstance(obj, list) else obj
            relationship_results[f"{subj_str} {direction} {obj_str}"] = is_correct

    if relationship_results:
        results["relationships"] = relationship_results
        results["all_relationships_correct"] = all(relationship_results.values()) if relationship_results else False
        results["overall_correct"] = results["all_objects_detected"] and results["all_relationships_correct"]
    else:
        results["overall_correct"] = results["all_objects_detected"]


    results["bboxes"] = bbox_dict
    results["counts"] = object_counts
    results["missing_objects"] = missing_objects
    results["excess_objects"] = excess_objects
    results["model_response"] = detection_response
    results["object_descriptions"] = object_descriptions
    results["model_prompt"] = detection_prompt

    return results


def evaluate_spatial_image(model, processor, image_path: str, task_info: Dict[str, Any]):
    objects = task_info.get('objects', [])
    colors = task_info.get('colors', [])
    relations = task_info.get('relative_position', [])


    has_nested = isinstance(objects, list) and any(isinstance(item, list) for item in objects)
    if has_nested:
        flat_objects = [o for group in objects for o in group]
        flat_colors = [c for group in colors for c in group] if colors and isinstance(colors[0], list) else colors
    else:
        flat_objects = objects
        flat_colors = colors

    colored_objects = prepare_colored_objects(flat_objects, flat_colors) if flat_colors else flat_objects


    processed_relations = []
    if relations:
        obj_to_colored: Dict[str, str] = {}
        if flat_colors:
            for i, obj in enumerate(flat_objects):
                if i < len(flat_colors) and flat_colors[i]:
                    obj_to_colored[obj] = f"{flat_colors[i]} {obj}"
        for rel in relations:
            pr = rel.copy()
            if isinstance(rel['subject'], str) and rel['subject'] in obj_to_colored:
                pr['subject'] = obj_to_colored[rel['subject']]
            elif isinstance(rel['subject'], list):
                pr['subject'] = [obj_to_colored.get(x, x) for x in rel['subject']]
            if isinstance(rel['object'], str) and rel['object'] in obj_to_colored:
                pr['object'] = obj_to_colored[rel['object']]
            elif isinstance(rel['object'], list):
                pr['object'] = [obj_to_colored.get(x, x) for x in rel['object']]
            processed_relations.append(pr)

    results = check_spatial_relation(
        model, processor, image_path, colored_objects, processed_relations, flat_colors
    )


    results['original_relations'] = relations
    results['objects'] = objects
    if has_nested:
        results['has_nested_objects'] = True


    return results




def process_tasks(model, processor, tasks: List[Dict[str, Any]], img_dir: str, output_base_dir: str):
    task_results: List[Dict[str, Any]] = []
    task_correct_count = 0
    total_spatial_relations = 0
    correct_spatial_relations = 0

    image_files = list_generated_images(model_dir:=os.path.dirname(img_dir), task_subdir=os.path.basename(img_dir))
    id_to_image_path: Dict[str, str] = {}
    for p in image_files:
        base = os.path.splitext(os.path.basename(p))[0]
        pid = parse_image_id(base) or base
        id_to_image_path[pid] = p

    for i, task in enumerate(tqdm(tasks, desc="Evaluating spatial tasks")):
        task_id = task.get("id", i + 1)
        image_key = parse_image_id(str(task_id)) or f"{int(task_id):03d}" if str(task_id).isdigit() else str(task_id)
        image_path = id_to_image_path.get(image_key)
        if not image_path or not os.path.exists(image_path):
            logging.warning(f"Image not found for id={task_id} under {img_dir}, skipping")
            continue

        result = evaluate_spatial_image(model, processor, image_path, task["task"])
        if 'relationships' in result:
            task_total_rel = len(result['relationships'])
            task_correct_rel = sum(1 for v in result['relationships'].values() if v)
            total_spatial_relations += task_total_rel
            correct_spatial_relations += task_correct_rel
            result['task_relation_accuracy'] = task_correct_rel / task_total_rel if task_total_rel > 0 else 0
        if result.get('overall_correct', False):
            task_correct_count += 1

        result['id'] = task_id
        result['prompt'] = task['prompt']
        task_results.append(result)




    task_match_rate = task_correct_count / len(task_results) if task_results else 0.0
    stats: Dict[str, Any] = {
        "task_match_rate": task_match_rate,
        "total_tasks": len(task_results),
        "relation_accuracy": (correct_spatial_relations / total_spatial_relations) if total_spatial_relations > 0 else 0.0,
        "total_relations": total_spatial_relations,
        "correct_relations": correct_spatial_relations,
        "total": len(task_results),
        "correct": task_correct_count,
        "accuracy": task_match_rate,
    }


    return task_results, stats


def evaluate_model_spatial(model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    print(f"Evaluating model {model_name} on SpatialLayout task ...")
    model_dir = os.path.join(args.models_dir, model_name)
    img_dir = os.path.join(model_dir, 'SpatialLayout')
    if not os.path.exists(img_dir):
        print(f"Warning: SpatialLayout images directory does not exist: {img_dir}")
        return {
            'task': 'SpatialLayout',
            'model': model_name,
            'stats': {
                'task_match_rate': 0.0,
                'total_tasks': 0,
                'relation_accuracy': 0.0,
                'total_relations': 0,
                'correct_relations': 0,
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
            }
        }

    eval_model, eval_processor, _ = load_model(
        args.internvl_model_path,
        device=args.device,
    )

    spatial_json = args.spatial_json or os.path.join(args.dataset_dir, 'benchmark', 'prompt', 'SpatialLayout.json')
    if not os.path.exists(spatial_json):
        alt = os.path.join(args.dataset_dir, 'prompt', 'SpatialLayout.json')
        spatial_json = alt if os.path.exists(alt) else spatial_json
    with open(spatial_json, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    model_output_dir = os.path.join(args.output_dir, model_name)
    ensure_dir(model_output_dir)

    task_results, stats = process_tasks(
        eval_model, eval_processor, tasks, img_dir, model_output_dir
    )

    model_summary = {
        'task': 'SpatialLayout',
        'model': model_name,
        'stats': stats,
    }

    write_json(os.path.join(model_output_dir, 'spatiallayout.json'), {"SpatialLayout": stats, "overall_score": stats.get('task_match_rate', 0.0)})

    print(f"Completed model {model_name} - SpatialLayout: {stats['correct']}/{stats['total']} correct")
    return model_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch evaluate models on Text-to-Image SpatialLayout task')
    parser.add_argument('--dataset_dir', default='', help='Dataset root directory')
    parser.add_argument('--models_dir', default='', help='Model outputs root directory (images)')
    parser.add_argument('--output_dir', default='', help='Evaluation outputs directory')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate')

    parser.add_argument('--internvl_model_path', type=str, default='OpenGVLab/InternVL3-14B', help='InternVL model path or hub id')
    parser.add_argument('--device', type=str, default='cuda', help='Device for evaluator (cuda/cpu)')



    parser.add_argument('--spatial_json', type=str, default=None, help='Path to SpatialLayout JSON (default: <dataset_dir>/benchmark/prompt/SpatialLayout.json)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, d))]

    print(f"Models to evaluate: {models}")
    print("Task: SpatialLayout")
    print("-" * 50)

    all_results: Dict[str, Any] = {}
    for model_name in models:
        model_summary = evaluate_model_spatial(model_name, args)
        all_results[model_name] = model_summary

    minified_results: Dict[str, float] = {}
    for model_name, summary in all_results.items():
        stats = summary.get('stats', {}) if isinstance(summary, dict) else {}
        minified_results[model_name] = float(stats.get('accuracy', 0.0))
    summary_json = os.path.join(args.output_dir, 'SpatialLayout_summary.json')
    write_json(summary_json, minified_results)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()


