import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from tqdm import tqdm
import traceback
import re
from paddleocr import PaddleOCR
from utils import ensure_dir, write_json


def extract_numbers_from_image(image_path: str, confidence_threshold: float) -> Tuple[List[Tuple[str, Tuple[int, int]]], List[Dict]]:
    ocr_model = PaddleOCR(
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=False,
        device="gpu",
    )
    print(f"Starting to process image {image_path}")
    result = ocr_model.predict(image_path)
    print(f"Prediction result type={type(result)}, length={len(result) if result else 0}")
    if result is None or len(result) == 0:
        print(f"Warning: No text detected in image {image_path}")
        return [], []
    extracted_data: List[Tuple[str, Tuple[int, int]]] = []
    ocr_details: List[Dict] = []
    for res in result:
        if isinstance(res, dict):
            rec_texts = res.get('rec_texts', [])
            rec_scores = res.get('rec_scores', [])
            rec_boxes = res.get('rec_boxes', [])
            print(f"Found {len(rec_texts)} texts, {len(rec_scores)} scores, {len(rec_boxes)} bounding boxes")
            if len(rec_texts) == len(rec_scores) == len(rec_boxes):
                for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                    if not text or text.strip() == '':
                        continue
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    ocr_details.append({
                        'text': text,
                        'confidence': float(score),
                        'position': (center_x, center_y),
                        'box': box.tolist() if hasattr(box, 'tolist') else list(box),
                        'is_high_confidence': score > confidence_threshold
                    })
                    if score > confidence_threshold:
                        numbers = re.findall(r'\d', text)
                        if numbers:
                            for num in numbers:
                                extracted_data.append((num, (center_x, center_y)))
            else:
                print(f"Warning: Text count({len(rec_texts)}), score count({len(rec_scores)}) and bbox count({len(rec_boxes)}) do not match")
                min_len = min(len(rec_texts), len(rec_scores), len(rec_boxes))
                for i in range(min_len):
                    text = rec_texts[i]
                    score = rec_scores[i] if i < len(rec_scores) else 1.0
                    box = rec_boxes[i] if i < len(rec_boxes) else [0, 0, 100, 100]
                    if text and text.strip() != '':
                        center_x = int((box[0] + box[2]) / 2)
                        center_y = int((box[1] + box[3]) / 2)
                        ocr_details.append({
                            'text': text,
                            'confidence': float(score),
                            'position': (center_x, center_y),
                            'box': box.tolist() if hasattr(box, 'tolist') else list(box),
                            'is_high_confidence': True
                        })
                        numbers = re.findall(r'\d', text)
                        for num in numbers:
                            extracted_data.append((num, (center_x, center_y)))
            break
        else:
            print(f"Unrecognized result structure: {type(res)}")
            continue
    print(f"Finally extracted {len(extracted_data)} valid numbers and positions")
    return extracted_data, ocr_details


def load_gt_sudoku_image(ground_truth_path: str) -> Optional[str]:
        gt_dir = os.path.dirname(ground_truth_path)                          
        dataset_dir = os.path.dirname(gt_dir)                   
        answer_dir = os.path.join(dataset_dir, "answer")                          


        filename = os.path.basename(ground_truth_path).replace('.json', '.png')
        gt_image_path = os.path.join(answer_dir, filename)

        if os.path.exists(gt_image_path):
            return gt_image_path
        else:
            print(f"Warning: GT image does not exist {gt_image_path}")
            return None

def load_question_sudoku_image(ground_truth_path: str) -> Optional[str]:
        gt_dir = os.path.dirname(ground_truth_path)
        dataset_dir = os.path.dirname(gt_dir)
        question_dir = os.path.join(dataset_dir, "question") 

        filename = os.path.basename(ground_truth_path).replace('.json', '.png')
        question_image_path = os.path.join(question_dir, filename)

        if os.path.exists(question_image_path):
            return question_image_path
        else:
            print(f"Warning: question image does not exist {question_image_path}")
            return None


def compute_distance_threshold(gt_data: List[Tuple[str, Tuple[int, int]]], grid_size: int) -> float:
    if gt_data:
        positions = [pos for _, pos in gt_data]
        if len(positions) >= 4:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            avg_cell_size = max(x_range, y_range) / (grid_size - 1) if grid_size > 1 else 100
            return max(avg_cell_size * 0.6, 60.0)
        return 80.0
    return 80.0


def calculate_accuracy_with_positions(predicted_data: List[Tuple[str, Tuple[int, int]]],
                                      gt_data: List[Tuple[str, Tuple[int, int]]],
                                      question_data: List[Tuple[str, Tuple[int, int]]],
                                      grid_size: int,
                                      distance_threshold: float = None) -> Dict[str, float]:
    if distance_threshold is None:
        distance_threshold = compute_distance_threshold(gt_data, grid_size)

    print(f"Using distance threshold: {distance_threshold:.1f} px")


    question_positions = {(qx, qy) for _, (qx, qy) in question_data}


    empty_positions: List[Tuple[str, Tuple[int, int]]] = []
    for gt_num, (gt_x, gt_y) in gt_data:
        is_original = any((((gt_x - qx) ** 2 + (gt_y - qy) ** 2) ** 0.5) < distance_threshold for (qx, qy) in question_positions)
        if not is_original:
            empty_positions.append((gt_num, (gt_x, gt_y)))

    total_empty_cells = len(empty_positions)
    matched_count = 0
    matched_empty_indices: set = set()
    prediction_analysis: List[Dict[str, Any]] = []


    for pred_idx, (pred_num, (pred_x, pred_y)) in enumerate(predicted_data):

        matches_original = False
        original_match_info = None
        for q_num, (q_x, q_y) in question_data:
            distance = ((pred_x - q_x) ** 2 + (pred_y - q_y) ** 2) ** 0.5
            if distance < distance_threshold:
                matches_original = True
                original_match_info = {
                    'question_num': q_num,
                    'is_correct': pred_num == q_num,
                    'distance': distance
                }
                break

        if matches_original:
            if original_match_info['is_correct']:
                prediction_analysis.append({
                    'type': 'original_correct', 'pred_num': pred_num, 'position': (pred_x, pred_y), 'contributes_to_total': False, 'is_correct': True
                })
            else:
                prediction_analysis.append({
                    'type': 'original_wrong', 'pred_num': pred_num, 'position': (pred_x, pred_y), 'contributes_to_total': True, 'is_correct': False
                })
            continue


        best_empty_match_idx = -1
        best_distance = float('inf')
        for i, (empty_num, (empty_x, empty_y)) in enumerate(empty_positions):
            if i in matched_empty_indices:
                continue
            distance = ((pred_x - empty_x) ** 2 + (pred_y - empty_y) ** 2) ** 0.5
            if pred_num == empty_num and distance < distance_threshold and distance < best_distance:
                best_empty_match_idx = i
                best_distance = distance
        if best_empty_match_idx != -1:
            matched_count += 1
            matched_empty_indices.add(best_empty_match_idx)
            prediction_analysis.append({
                'type': 'empty_correct', 'pred_num': pred_num, 'position': (pred_x, pred_y), 'contributes_to_total': False, 'is_correct': True
            })
        else:
            prediction_analysis.append({
                'type': 'no_match', 'pred_num': pred_num, 'position': (pred_x, pred_y), 'contributes_to_total': False, 'is_correct': False
            })

    wrong_original_count = sum(1 for p in prediction_analysis if p['type'] == 'original_wrong')
    final_total = total_empty_cells + wrong_original_count

    position_accuracy = matched_count / final_total if final_total > 0 else 0.0

    return {
        'position_accuracy': position_accuracy,
        'prediction_analysis': prediction_analysis,
        'distance_threshold': float(distance_threshold),
    }





def evaluate_single_sudoku_image(predicted_path: str, ground_truth_path: str, 
                                confidence_threshold: float = 0.5) -> Optional[Dict]:
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    grid_size = ground_truth_data['grid_size']

    gt_image_path = load_gt_sudoku_image(ground_truth_path)
    question_image_path = load_question_sudoku_image(ground_truth_path)

    predicted_data, _ = extract_numbers_from_image(predicted_path, confidence_threshold)
    gt_data: List[Tuple[str, Tuple[int, int]]] = []
    if gt_image_path:
        gt_data, _ = extract_numbers_from_image(gt_image_path, confidence_threshold)
    question_data: List[Tuple[str, Tuple[int, int]]] = []
    if question_image_path:
        question_data, _ = extract_numbers_from_image(question_image_path, confidence_threshold)

    position_metrics = calculate_accuracy_with_positions(
        predicted_data, gt_data, question_data, grid_size
    )
    return position_metrics



def get_sudoku_file_mapping(dataset_dir: str, model_dir: str) -> List[Tuple[str, str, str]]:
    file_mappings: List[Tuple[str, str, str]] = []
    dataset_task_dir = os.path.join(dataset_dir, "VisualLogic", "sudoku")
    model_task_dir = os.path.join(model_dir, "VisualLogic")
    if not os.path.exists(dataset_task_dir) or not os.path.exists(model_task_dir):
        return file_mappings
    for pred_file in os.listdir(model_task_dir):
        if pred_file.endswith('.png'):
            image_id = pred_file[:-4]
            ground_truth_path = os.path.join(dataset_task_dir, f"{image_id}.json")
            predicted_path = os.path.join(model_task_dir, pred_file)
            if os.path.exists(ground_truth_path) and os.path.exists(predicted_path):
                file_mappings.append((image_id, predicted_path, ground_truth_path))
    return file_mappings


def evaluate_model_sudoku(model_name: str, dataset_dir: str, model_dir: str, 
                         output_dir: str, confidence_threshold: float = 0.5) -> Dict:
    print(f"Evaluating model {model_name} on Sudoku task...")

    file_mappings = get_sudoku_file_mapping(dataset_dir, model_dir)

    if not file_mappings:
        print(f"Warning: No valid files found for {model_name} in Sudoku task")
        return {
            'task': 'Sudoku',
            'model': model_name,
            'total_images': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_metrics': {}
        }


    model_output_dir = os.path.join(output_dir, model_name)
    ensure_dir(model_output_dir)


    all_results = []
    successful_count = 0
    failed_count = 0

    for image_id, predicted_path, ground_truth_path in tqdm(file_mappings, 
                                                           desc="Evaluating Sudoku"):
        result = evaluate_single_sudoku_image(predicted_path, ground_truth_path, confidence_threshold)
        if result is None:
            failed_count += 1
            continue
        result['image_id'] = image_id
        all_results.append(result)
        successful_count += 1


    if all_results:
        acc_values = [r.get('position_accuracy', 0.0) for r in all_results if 'position_accuracy' in r]
        average_metrics = {'avg_position_accuracy': float(np.mean(acc_values))} if acc_values else {}
    else:
        average_metrics = {}


    task_results = {
        'task': 'Sudoku',
        'model': model_name,
        'total_images': len(file_mappings),
        'successful_evaluations': successful_count,
        'failed_evaluations': failed_count,
        'success_rate': successful_count / len(file_mappings) if file_mappings else 0.0,
        'average_metrics': average_metrics,
        'confidence_threshold': confidence_threshold
    }

    task_summary_path = os.path.join(model_output_dir, "sudoku.json")
    write_json(task_summary_path, task_results)

    print(f"Completed evaluation {model_name} - Sudoku: success {successful_count}/{len(file_mappings)} images")
    print(f"Task results saved to: {task_summary_path}")

    return task_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch evaluate models on Sudoku task')
    parser.add_argument('--dataset_dir', 
                       default='',
                       help='Dataset root directory')
    parser.add_argument('--models_dir', 
                       default='',
                       help='Model output root directory')
    parser.add_argument('--output_dir', 
                       default='',
                       help='Evaluation output directory')
    parser.add_argument('--models', 
                       nargs='+',
                       help='List of models to evaluate (default: all)')
    parser.add_argument('--confidence_threshold', 
                       type=float, 
                       default=0.5, 
                       help='OCR confidence threshold')

    args = parser.parse_args()


    ensure_dir(args.output_dir)


    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(args.models_dir) 
                 if os.path.isdir(os.path.join(args.models_dir, d))]

    print(f"Models to evaluate: {models}")
    print(f"Evaluation task: Sudoku")
    print(f"OCR confidence threshold: {args.confidence_threshold}")
    print("-" * 50)


    all_results = {}

    for model_name in models:
        print(f"\nStart evaluating model: {model_name}")
        model_dir = os.path.join(args.models_dir, model_name)
        if not os.path.exists(model_dir):
            print(f"Warning: model directory does not exist {model_dir}")
            continue
        model_results = {}
        task_result = evaluate_model_sudoku(
            model_name=model_name,
            dataset_dir=args.dataset_dir,
            model_dir=model_dir,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence_threshold
        )
        model_results["sudoku"] = task_result
        all_results[model_name] = model_results


    minified = {}
    for model_name, model_results in all_results.items():
        task_result = next(iter(model_results.values())) if model_results else {}
        avg_acc = (task_result or {}).get('average_metrics', {}).get('avg_position_accuracy', 0.0)
        minified[model_name] = float(avg_acc)
    summary_path = os.path.join(args.output_dir, "sudoku_summary.json")
    write_json(summary_path, minified)

    print(f"\nEvaluation completed! Overall results saved to: {summary_path}")


if __name__ == "__main__":
    main()
