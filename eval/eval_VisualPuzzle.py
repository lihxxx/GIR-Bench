import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from PIL import Image
from utils import ensure_dir, write_json
from torchmetrics.image.fid import FrechetInceptionDistance


def load_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)


def initialize_fid_metric(device: str, feature_dim: int = 2048) -> FrechetInceptionDistance:
    return FrechetInceptionDistance(feature=feature_dim).to(device)


def fid_update_image(fid_metric: FrechetInceptionDistance, device: str, image: np.ndarray, real: bool) -> None:
    img = Image.fromarray(image).convert('RGB').resize((512, 512))
    img_array = np.array(img)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor.clamp(0, 255).to(torch.uint8)
    fid_metric.update(img_tensor, real=real)


def normalize_fid(fid_value: float, max_fid: float = 300.0, normalization_type: str = "exp") -> float:
    if fid_value < 0:
        fid_value = 0.0
    if normalization_type == "linear":
        normalized = 1.0 - min(fid_value / max_fid, 1.0)
    elif normalization_type == "exp":
        scale = max_fid / 5.0
        normalized = float(np.exp(-fid_value / scale))
    elif normalization_type == "sigmoid":
        midpoint = max_fid / 2.0
        scale = max_fid / 10.0
        normalized = float(1.0 / (1.0 + np.exp((fid_value - midpoint) / scale)))
    else:
        scale = max_fid / 5.0
        normalized = float(np.exp(-fid_value / scale))
    return float(np.clip(normalized, 0.0, 1.0))


def get_jigsaw_file_mapping(dataset_dir: str, model_dir: str) -> List[Tuple[str, str, str]]:
    file_mappings: List[Tuple[str, str, str]] = []
    dataset_task_dir = os.path.join(dataset_dir, "VisualPuzzle")
    model_task_dir = os.path.join(model_dir, "VisualPuzzle")
    if not os.path.exists(dataset_task_dir) or not os.path.exists(model_task_dir):
        return file_mappings
    for pred_file in os.listdir(model_task_dir):
        if pred_file.endswith('.png'):
            image_id = pred_file[:-4]
            solution_path = os.path.join(dataset_task_dir, "solution", f"{image_id.zfill(3)}.png")
            predicted_path = os.path.join(model_task_dir, pred_file)
            if os.path.exists(solution_path) and os.path.exists(predicted_path):
                file_mappings.append((image_id, predicted_path, solution_path))
    return file_mappings


def evaluate_model_jigsaw(model_name: str, dataset_dir: str, model_dir: str, 
                         output_dir: str, device: str = "cuda") -> Dict:
    print(f"Evaluating model {model_name} performance on VisualPuzzle task...")
    fid_metric = initialize_fid_metric(device)
    file_mappings = get_jigsaw_file_mapping(dataset_dir, model_dir)
    if not file_mappings:
        return {
            'task': 'VisualPuzzle',
            'model': model_name,
            'total_images': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_metrics': {},
            'fid': 0.0,
            'normalized_fid': 0.0
        }
    model_output_dir = os.path.join(output_dir, model_name)
    ensure_dir(model_output_dir)


    successful_count = 0
    failed_count = 0

    for image_id, predicted_path, solution_path in tqdm(file_mappings, desc="Evaluating Jigsaw"):
        pred_img = load_image(predicted_path)
        gt_img = load_image(solution_path)
        fid_update_image(fid_metric, device, gt_img, real=True)
        fid_update_image(fid_metric, device, pred_img, real=False)
        successful_count += 1


    fid_value = float(fid_metric.compute())
    normalized_fid = normalize_fid(fid_value, max_fid=300.0, normalization_type="exp")

    task_results = {
        'task': 'VisualPuzzle',
        'model': model_name,
        'total_images': len(file_mappings),
        'successful_evaluations': successful_count,
        'failed_evaluations': failed_count,
        'success_rate': successful_count / len(file_mappings) if file_mappings else 0.0,
        'average_metrics': {},
        'fid': float(fid_value),
        'normalized_fid': float(normalized_fid),
        'device': device
    }

    task_summary_path = os.path.join(model_output_dir, "visualpuzzle.json")
    write_json(task_summary_path, task_results)
    print(f"Completed evaluation {model_name} - VisualPuzzle: Success {successful_count}/{len(file_mappings)} images")
    print(f"Task results saved to: {task_summary_path}")
    return task_results


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate models on VisualPuzzle task (FID only)')
    parser.add_argument('--dataset_dir', 
                       default='',
                       help='Dataset root directory')
    parser.add_argument('--models_dir', 
                       default='',
                       help='Model output root directory')
    parser.add_argument('--output_dir', 
                       default='',
                       help='Evaluation results output directory')
    parser.add_argument('--models', 
                       nargs='+',
                       help='List of models to evaluate (default: evaluate all models)')
    parser.add_argument('--device', 
                       type=str, 
                       default='cuda', 
                       help='Computing device (default: cuda)')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(args.models_dir) 
                 if os.path.isdir(os.path.join(args.models_dir, d))]

    print(f"Models to evaluate: {models}")
    print(f"Computing device: {args.device}")
    print("-" * 50)

    all_results = {}
    for model_name in models:
        print(f"\nStart evaluating model: {model_name}")
        model_dir = os.path.join(args.models_dir, model_name)
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory does not exist {model_dir}")
            continue
        model_results = {}
        task_result = evaluate_model_jigsaw(
            model_name=model_name,
            dataset_dir=args.dataset_dir,
            model_dir=model_dir,
            output_dir=args.output_dir,
            device=args.device
        )
        model_results["VisualPuzzle"] = task_result
        all_results[model_name] = model_results

    minified = {}
    for model_name, model_results in all_results.items():
        task_result = next(iter(model_results.values())) if model_results else {}
        if task_result:
            minified[model_name] = float(task_result.get('normalized_fid', 0.0))
    summary_path = os.path.join(args.output_dir, "visualpuzzle_summary.json")
    write_json(summary_path, minified)
    print(f"\nEvaluation completed! Overall results saved to: {summary_path}")


if __name__ == "__main__":
    main()
