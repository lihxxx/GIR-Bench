import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple
from utils import ensure_dir, write_json, NumpyEncoder, resolve_models

import numpy as np
from PIL import Image
from tqdm import tqdm
from paddleocr import PaddleOCR


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lcs_length_word(words1: List[str], words2: List[str]) -> int:
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1].lower() == words2[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def lcs_score(gt: str, pred: str) -> float:
    if not gt or not pred:
        return 0.0
    gt_words = gt.split()
    pred_words = pred.split()
    if not gt_words:
        return 1.0 if not pred_words else 0.0
    lcs = lcs_length_word(gt_words, pred_words)
    return lcs / len(gt_words)


def find_word_aligned_continuous_matches(gt: str, pred: str) -> int:
    if not gt or not pred:
        return 0
    gt_normalized = normalize_text(gt)
    pred_normalized = normalize_text(pred)
    if not gt_normalized:
        return 0
    gt_words = gt_normalized.split()
    word_positions = []
    current_pos = 0
    for word in gt_words:
        start_pos = gt_normalized.find(word, current_pos)
        if start_pos != -1:
            end_pos = start_pos + len(word)
            word_positions.append({'word': word, 'start': start_pos, 'end': end_pos})
            current_pos = end_pos
    char_matches = []
    m, n = len(gt_normalized), len(pred_normalized)
    for i in range(m):
        for j in range(n):
            length = 0
            while (i + length < m and j + length < n and gt_normalized[i + length] == pred_normalized[j + length]):
                length += 1
            if length > 0:
                char_matches.append({'start_gt': i, 'end_gt': i + length, 'length': length})
    if not char_matches:
        return 0
    matched_words = set()
    for match in char_matches:
        for wi in word_positions:
            if match['start_gt'] <= wi['start'] and match['end_gt'] >= wi['end']:
                matched_words.add(wi['word'])
    return len(matched_words)


def lcs_continuous_word_score(gt: str, pred: str) -> float:
    if not gt or not pred:
        return 0.0
    gt_normalized = normalize_text(gt)
    pred_normalized = normalize_text(pred)
    if not gt_normalized:
        return 1.0 if not pred_normalized else 0.0
    gt_words = gt_normalized.split()
    if not gt_words:
        return 1.0 if not pred_normalized else 0.0
    matched_word_count = find_word_aligned_continuous_matches(gt, pred)
    return matched_word_count / len(gt_words)


def calculate_text(predicted_text: str, ground_truth: str) -> Dict[str, float]:
    pred_normalized = normalize_text(predicted_text)
    gt_normalized = normalize_text(ground_truth)
    lcs_continuous_similarity = lcs_continuous_word_score(gt_normalized, pred_normalized)
    return {
        'lcs_continuous_score': lcs_continuous_similarity,
    }



def load_ocr_model(device: str = 'gpu') -> PaddleOCR:
    print("Initializing PaddleOCR model...")
    model = PaddleOCR(
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu" if device == 'cuda' or device == 'gpu' else 'cpu',
    )
    print("PaddleOCR initialized")
    return model


def extract_text_from_image(ocr_model: PaddleOCR,
                           image_path: str,
                           confidence_threshold: float = 0.5) -> Tuple[str, str]:
    print(f"Processing image {image_path}")
    result = ocr_model.predict(image_path)
    print(f"Prediction result type={type(result)}, len={len(result) if result else 0}")
    if result is None or len(result) == 0:
        print(f"Warning: no text detected in {image_path}")
        return "", "[]"
    extracted_texts: List[str] = []
    result_details: List[Dict[str, Any]] = []
    for res in result:
        if isinstance(res, dict):
            rec_texts = res.get('rec_texts', [])
            rec_scores = res.get('rec_scores', [])
            print(f"Found {len(rec_texts)} texts, {len(rec_scores)} scores")
            if len(rec_texts) == len(rec_scores):
                for text, score in zip(rec_texts, rec_scores):
                    if not text or text.strip() == '':
                        continue
                    if score > confidence_threshold:
                        extracted_texts.append(text)
                    result_details.append({'text': text, 'confidence': float(score)})
            else:
                print(f"Warning: text count({len(rec_texts)}) != score count({len(rec_scores)})")
                for text in rec_texts:
                    if text and text.strip() != '':
                        extracted_texts.append(text)
                        result_details.append({'text': text, 'confidence': 1.0})
            break
    extracted_text = ' '.join(extracted_texts)
    print(f"Extracted {len(extracted_texts)} texts: '{extracted_text}'")
    return extracted_text, str(result_details)


def evaluate_text_rendering_image(ocr_model: PaddleOCR,
                                  image_path: str,
                                  task_info: Dict[str, Any],
                                  confidence_threshold: float = 0.5) -> Dict[str, Any]:
    ground_truth = task_info["text"]
    predicted_text, ocr_result = extract_text_from_image(
        ocr_model, image_path, confidence_threshold
    )
    similarity = calculate_text(predicted_text, ground_truth)
    return {
        "ground_truth": ground_truth,
        "predicted_text": predicted_text,
        "ocr_result": str(ocr_result),
        "lcs_continuous_score": similarity.get('lcs_continuous_score', 0.0)
    }



def process_text_tasks(ocr_model: PaddleOCR,
                       tasks: List[Dict[str, Any]],
                       img_dir: str,
                       output_dir: str,
                       confidence_threshold: float = 0.5,
                       skip_existing: bool = False):
    ensure_dir(output_dir)

    task_results: List[Dict[str, Any]] = []
    existing_results: Dict[Any, Dict[str, Any]] = {}

    for i, task in enumerate(tqdm(tasks, desc="Evaluating text rendering tasks")):
        task_id = task.get("id", i + 1)
        image_filename = f"{task_id:03d}.png"
        image_path = os.path.join(img_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Warning: image not found {image_path}, skip")
            continue
        if skip_existing and task_id in existing_results:
            task_results.append(existing_results[task_id])
            print(f"✓ Loaded cached text task {task_id}")
            continue

        result = evaluate_text_rendering_image(
            ocr_model, image_path, task, confidence_threshold
        )
        result["id"] = task_id
        result["prompt"] = task["prompt"]
        task_results.append(result)
        print(f"Processed text task {task_id}: lcs_continuous_score {result.get('lcs_continuous_score', 0.0):.4f}")

    if task_results:
        total_tasks = len(task_results)
        avg_lcs_continuous_score = sum(r.get('lcs_continuous_score', 0.0) for r in task_results) / total_tasks
        stats = {
            "total_tasks": total_tasks,
            "avg_lcs_continuous_score": avg_lcs_continuous_score,
        }
    else:
        stats = {
            "total_tasks": 0,
            "avg_lcs_continuous_score": 0.0,
        }

    return task_results, stats


def evaluate_model_text(model_name: str,
                        args: argparse.Namespace,
                        ocr_model: PaddleOCR) -> Dict[str, Any]:
    print(f"Evaluating model {model_name} on TextRendering ...")
    model_dir = os.path.join(args.models_dir, model_name)
    img_dir = os.path.join(model_dir, 'TextRendering')
    if not os.path.exists(img_dir):
        print(f"Warning: TextRendering image directory does not exist: {img_dir}")
        return {
            'task': 'TextRendering',
            'model': model_name,
            'stats': {
                'total_tasks': 0,
                'exact_match_count': 0,
                'exact_match_rate': 0.0,
                'avg_score': 0.0,
                'avg_lcs_score': 0.0,
                'avg_edit_distance_score': 0.0,
                'avg_lcs_continuous_score': 0.0,
                'avg_overall_score': 0.0,
            }
        }


    text_json = args.text_json
    if not os.path.exists(text_json):
        alt = os.path.join(args.dataset_dir, 'prompt', 'TextRendering.json')
        text_json = alt if os.path.exists(alt) else text_json
    with open(text_json, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    model_output_dir = os.path.join(args.output_dir, model_name)
    ensure_dir(model_output_dir)

    task_results, stats = process_text_tasks(
        ocr_model, tasks, img_dir, model_output_dir,
        confidence_threshold=args.confidence_threshold,
        skip_existing=args.skip_existing,
    )

    model_summary = {
        'task': 'TextRendering',
        'model': model_name,
        'stats': stats,
    }
    write_json(os.path.join(model_output_dir, 'textrendering.json'), {"TextRendering": stats, "avg_lcs_continuous_score": stats.get('avg_lcs_continuous_score', 0.0)})

    print(f"Completed {model_name} - text: avg_lcs_continuous_score={stats.get('avg_lcs_continuous_score', 0.0):.4f} over {stats.get('total_tasks', 0)} tasks")
    return model_summary



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch evaluate models on Text-to-Image TextRendering task (PaddleOCR)')
    parser.add_argument('--dataset_dir', default='', help='Dataset root directory')
    parser.add_argument('--models_dir', default='', help='Model outputs root directory (images)')
    parser.add_argument('--output_dir', default='', help='Evaluation outputs directory')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate (default: all subdirectories in models_dir)')
    parser.add_argument('--text_json', type=str, default='', help='Path to TextRendering task JSON')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='OCR confidence threshold (0.0-1.0)')
    parser.add_argument('--skip_existing', action='store_true', help='Skip tasks that already have outputs')
    parser.add_argument('--device', type=str, default='cuda', help='Device for PaddleOCR (cuda/cpu)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    models = resolve_models(args.models_dir, args.models)
    print(f"Models to evaluate: {models}")
    print("Task: TextRendering (PaddleOCR)")
    print("-" * 50)


    ocr_model = load_ocr_model(device=args.device)

    all_results: Dict[str, Any] = {}
    for model_name in models:
        summary = evaluate_model_text(model_name, args, ocr_model)
        all_results[model_name] = summary


    minified: Dict[str, float] = {}
    for model_name, summary in all_results.items():
        stats = summary.get('stats', {}) if isinstance(summary, dict) else {}
        minified[model_name] = float(stats.get('avg_lcs_continuous_score', 0.0))

    summary_json = os.path.join(args.output_dir, 'TextRendering_summary.json')
    write_json(summary_json, minified)


if __name__ == '__main__':
    main()


