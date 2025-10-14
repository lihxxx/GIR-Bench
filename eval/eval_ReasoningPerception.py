import os
import csv
import argparse
from utils import ensure_dir, write_json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from tqdm import tqdm
import traceback
from skimage.metrics import structural_similarity as ssim


def get_reasonseg_file_mapping(dataset_dir: str, model_dir: str) -> List[Tuple[str, str, str, str]]:
    file_mappings: List[Tuple[str, str, str, str]] = []
    dataset_task_dir = os.path.join(dataset_dir, "ReasoningPerception")
    model_task_dir = os.path.join(model_dir, "ReasoningPerception")
    if not os.path.exists(dataset_task_dir) or not os.path.exists(model_task_dir):
        return file_mappings
    for pred_file in os.listdir(model_task_dir):
        if pred_file.endswith('.png'):
            image_id = pred_file[:-4]
            input_path = os.path.join(dataset_task_dir, "image", f"{image_id}.png")
            mask_path = os.path.join(dataset_task_dir, "mask", f"{image_id}.png")
            predicted_path = os.path.join(model_task_dir, pred_file)
            if os.path.exists(input_path) and os.path.exists(mask_path) and os.path.exists(predicted_path):
                file_mappings.append((image_id, input_path, mask_path, predicted_path))
    return file_mappings

def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def resize_to_target(img: np.ndarray, target_shape: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    if len(img.shape) == 3:
        current_shape = img.shape[:2]
    else:
        current_shape = img.shape

    if current_shape == target_shape:
        return img


    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    if len(img.shape) == 3:
        resized = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=interpolation)
    else:
        resized = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=interpolation)

    return resized


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    flood = m.copy()
    ff_mask = np.zeros((h+2, w+2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0,0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(m, holes)


def morph_close(mask: np.ndarray, k: int = 5, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)*255
    if k <= 1:
        return m
    kernel = cv2.getStructuringElement(shape, (k,k))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


def morph_open(mask: np.ndarray, k: int = 3, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)*255
    if k <= 1:
        return m
    kernel = cv2.getStructuringElement(shape, (k,k))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)



def edge_stop(image_bgr: np.ndarray, canny1: int = 50, canny2: int = 120, dilate_k: int = 3) -> np.ndarray:
    edges = cv2.Canny(image_bgr, canny1, canny2)
    if dilate_k>1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k,dilate_k))
        edges = cv2.dilate(edges, kernel)
    return edges > 0




def detect_by_channel_ratio(orig_bgr: np.ndarray, mod_bgr: np.ndarray,
                           g_over_r: float = 1.15, g_over_b: float = 1.15,
                           delta_g: int = 10) -> np.ndarray:
    b0, g0, r0 = cv2.split(orig_bgr)
    b1, g1, r1 = cv2.split(mod_bgr)

    eps = 1e-6
    cond_mod = (g1.astype(np.float32)/(r1.astype(np.float32)+eps) > g_over_r) &\
               (g1.astype(np.float32)/(b1.astype(np.float32)+eps) > g_over_b)
    cond_delta = (g1.astype(np.int16) - g0.astype(np.int16)) > delta_g

    mask = np.where(cond_mod & cond_delta, 255, 0).astype(np.uint8)
    return mask


def clean_seeds_with_component_metrics(orig_bgr: np.ndarray, mod_bgr: np.ndarray, ratio_mask: np.ndarray,
                                      min_area: int = 200, min_mean_delta_g: int = 15, min_mean_s: int = 60,
                                      open_k: int = 3, close_k: int = 5, median_k: int = 0) -> np.ndarray:
    m = (ratio_mask > 0).astype(np.uint8)*255
    if median_k and median_k >=3 and median_k%2==1:
        m = cv2.medianBlur(m, median_k)
    if open_k>1:
        m = morph_open(m, open_k)
    if close_k>1:
        m = morph_close(m, close_k)

    hsv_mod = cv2.cvtColor(mod_bgr, cv2.COLOR_BGR2HSV)
    b0, g0, r0 = cv2.split(orig_bgr)
    b1, g1, r1 = cv2.split(mod_bgr)
    delta_g = g1.astype(np.int16) - g0.astype(np.int16)
    S = hsv_mod[:, :, 1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((m>0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area: continue
        ys, xs = np.where(labels==i)
        mean_dg = float(delta_g[ys, xs].mean()) if ys.size else 0.0
        mean_s  = float(S[ys, xs].mean()) if ys.size else 0.0
        if mean_dg >= min_mean_delta_g and mean_s >= min_mean_s:
            out[labels==i] = 255
    return out




def expand_with_grabcut(img_bgr: np.ndarray, seed_mask: np.ndarray,
                       fg_erode: int = 3, bg_erode: int = 3, iters: int = 5) -> np.ndarray:
    h, w = seed_mask.shape[:2]
    trimap = np.full((h,w), cv2.GC_PR_BGD, dtype=np.uint8)                                

    seed = (seed_mask>0).astype(np.uint8)*255
    if fg_erode>1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_erode, fg_erode))
        sure_fg = cv2.erode(seed, kernel)
    else:
        sure_fg = seed.copy()

    inv = cv2.bitwise_not(seed)
    if bg_erode>1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_erode, bg_erode))
        sure_bg = cv2.erode(inv, kernel)
    else:
        sure_bg = inv

    trimap[sure_bg>0] = cv2.GC_BGD
    trimap[sure_fg>0] = cv2.GC_FGD

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img_bgr, trimap, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    result = np.where((trimap==cv2.GC_FGD)|(trimap==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    result = morph_close(result, 3)
    result = fill_holes(result)
    return result




def expand_with_region_grow(img_bgr: np.ndarray, seed_mask: np.ndarray,
                           lab_sigma: float = 6.0,                                      
                           max_iters: int = 5,                            
                           edge_block: bool = True) -> np.ndarray:

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, A, B = cv2.split(lab)

    seeds = (seed_mask>0).astype(np.uint8)
    if seeds.sum()==0:
        return seeds*255


    muL, muA, muB = L[seeds>0].mean(), A[seeds>0].mean(), B[seeds>0].mean()

    def color_dist_sq(idx):
        l = L[idx]; a = A[idx]; b = B[idx]
        return (l-muL)**2 + (a-muA)**2 + (b-muB)**2

    edge_mask = edge_stop(img_bgr) if edge_block else None

    region = seeds.copy().astype(bool)
    H, W = region.shape
    for _ in range(max_iters):
        grown = region.copy()

        dil = cv2.dilate(region.astype(np.uint8)*255, np.ones((3,3), np.uint8))>0
        candidates = np.logical_and(dil, ~region)
        if edge_mask is not None:
            candidates = np.logical_and(candidates, ~edge_mask)
        idx = np.where(candidates)
        if len(idx[0])==0:
            break
        dist = color_dist_sq(idx)
        accept = dist <= (lab_sigma**2)
        acc_idx = (idx[0][accept], idx[1][accept])
        grown[acc_idx] = True

        if accept.sum() > 0:
            muL = (muL*region.sum() + L[acc_idx].sum()) / (region.sum() + accept.sum())
            muA = (muA*region.sum() + A[acc_idx].sum()) / (region.sum() + accept.sum())
            muB = (muB*region.sum() + B[acc_idx].sum()) / (region.sum() + accept.sum())

        if grown.sum() == region.sum():
            break
        region = grown

    out = (region.astype(np.uint8)*255)
    out = morph_close(out, 3)
    out = fill_holes(out)
    return out


def extract_edit_region_by_channel_ratio(input_img: np.ndarray, predicted_img: np.ndarray,
                                        g_over_r: float = 1.15, g_over_b: float = 1.15,
                                        delta_g: int = 10, min_area: int = 200,
                                        min_mean_delta_g: int = 15, min_mean_s: int = 60,
                                        open_k: int = 3, close_k: int = 5, median_k: int = 0,
                                        expand_method: str = "regiongrow", 
                                        lab_sigma: float = 6.0, grow_iters: int = 5,
                                        fg_erode: int = 3, bg_erode: int = 3, grab_iters: int = 5) -> np.ndarray:

    orig_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    mod_bgr = cv2.cvtColor(predicted_img, cv2.COLOR_RGB2BGR)


    ratio_raw = detect_by_channel_ratio(orig_bgr, mod_bgr, g_over_r, g_over_b, delta_g)


    seeds = clean_seeds_with_component_metrics(
        orig_bgr, mod_bgr, ratio_raw,
        min_area=min_area,
        min_mean_delta_g=min_mean_delta_g,
        min_mean_s=min_mean_s,
        open_k=open_k, close_k=close_k, median_k=median_k
    )


    if expand_method == "grabcut":
        expanded = expand_with_grabcut(
            mod_bgr, seeds,
            fg_erode=fg_erode,
            bg_erode=bg_erode,
            iters=grab_iters
        )
    else:              
        expanded = expand_with_region_grow(
            mod_bgr, seeds,
            lab_sigma=lab_sigma,
            max_iters=grow_iters,
            edge_block=True
        )

    return expanded




def extract_edit_region_by_color_enhanced(input_img: np.ndarray, predicted_img: np.ndarray, 
                                         threshold: int = 50, sensitivity: str = "medium") -> np.ndarray:
    predicted_bgr = cv2.cvtColor(predicted_img, cv2.COLOR_RGB2BGR)


    lower_green_bgr1 = np.array([0, 255 - threshold, 0], dtype=np.uint8)
    upper_green_bgr1 = np.array([threshold, 255, threshold], dtype=np.uint8)
    green_mask_bgr1 = cv2.inRange(predicted_bgr, lower_green_bgr1, upper_green_bgr1)


    target_green_bgr = np.array([87, 159, 123], dtype=np.uint8)
    lower_green_bgr2 = np.clip(target_green_bgr - threshold, 0, 255).astype(np.uint8)
    upper_green_bgr2 = np.clip(target_green_bgr + threshold, 0, 255).astype(np.uint8)
    green_mask_bgr2 = cv2.inRange(predicted_bgr, lower_green_bgr2, upper_green_bgr2)


    green_mask_bgr = cv2.bitwise_or(green_mask_bgr1, green_mask_bgr2)


    kernel_small = np.ones((3, 3), np.uint8)
    green_mask_bgr = cv2.morphologyEx(green_mask_bgr, cv2.MORPH_OPEN, kernel_small)

    kernel_medium = np.ones((5, 5), np.uint8)
    green_mask_bgr = cv2.morphologyEx(green_mask_bgr, cv2.MORPH_CLOSE, kernel_medium)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask_bgr, connectivity=8)

    total_area = green_mask_bgr.shape[0] * green_mask_bgr.shape[1]
    min_component_area = max(50, total_area * 0.0005)

    filtered_mask = np.zeros_like(green_mask_bgr)
    for i in range(1, num_labels):
        component_area = stats[i, cv2.CC_STAT_AREA]
        if component_area >= min_component_area:
            filtered_mask[labels == i] = 255

    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_small)

    return filtered_mask


def filter_small_regions(mask: np.ndarray, min_region_ratio: float = 0.03) -> np.ndarray:
    if min_region_ratio <= 0:
        return mask


    total_area = mask.shape[0] * mask.shape[1]
    min_component_area = int(total_area * min_region_ratio)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8)


    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):                             
        component_area = stats[i, cv2.CC_STAT_AREA]
        if component_area >= min_component_area:
            filtered_mask[labels == i] = 255

    return filtered_mask


def extract_edit_region(input_img: np.ndarray, predicted_img: np.ndarray, 
                       color_threshold: int = 50, mode: str = "fallback", 
                       color_sensitivity: str = "medium",
                       g_over_r: float = 1.15, g_over_b: float = 1.15, delta_g: int = 10,
                       expand_method: str = "regiongrow", lab_sigma: float = 6.0,
                       min_region_ratio: float = 0.03, return_method_info: bool = False, **kwargs):
    def is_mask_empty(mask: np.ndarray) -> bool:
        return np.sum(mask > 127) == 0

    method_info = {
        'used_method': mode,
        'ratio_used': False,
        'color_used': False,
        'color_empty': False
    }


    color_mask = extract_edit_region_by_color_enhanced(input_img, predicted_img, color_threshold, color_sensitivity)
    method_info['color_used'] = True

    if not is_mask_empty(color_mask):
        method_info['color_empty'] = False
        filtered_result = filter_small_regions(color_mask, min_region_ratio)
        return (filtered_result, method_info) if return_method_info else filtered_result

    method_info['color_empty'] = True
    method_info['ratio_used'] = True
    result = extract_edit_region_by_channel_ratio(input_img, predicted_img, 
                                                 g_over_r=g_over_r, g_over_b=g_over_b, delta_g=delta_g,
                                                 expand_method=expand_method, lab_sigma=lab_sigma)
    filtered_result = filter_small_regions(result, min_region_ratio)
    return (filtered_result, method_info) if return_method_info else filtered_result


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray, target_value: int = 255) -> float:

    binary_mask1 = (mask1 == target_value).astype(np.uint8)
    binary_mask2 = (mask2 == target_value).astype(np.uint8)


    intersection = np.logical_and(binary_mask1, binary_mask2).sum()
    union = np.logical_or(binary_mask1, binary_mask2).sum()


    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def calculate_precision_recall(predicted_mask: np.ndarray, gt_mask: np.ndarray, 
                              target_value: int = 255) -> Tuple[float, float]:

    pred_binary = (predicted_mask == target_value).astype(np.uint8)
    gt_binary = (gt_mask == target_value).astype(np.uint8)


    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary.astype(bool)).sum()
    fn = np.logical_and(~pred_binary.astype(bool), gt_binary).sum()


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall



def calculate_simple_background_ssim(input_img: np.ndarray, predicted_img: np.ndarray, 
                                   gt_mask: np.ndarray) -> float:
    background_mask = gt_mask <= 127
    if not np.any(background_mask):
        return 1.0
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    predicted_gray = cv2.cvtColor(predicted_img, cv2.COLOR_RGB2GRAY)
    input_bg = np.zeros_like(input_gray)
    predicted_bg = np.zeros_like(predicted_gray)
    input_bg[background_mask] = input_gray[background_mask]
    predicted_bg[background_mask] = predicted_gray[background_mask]
    background_ssim = ssim(input_bg, predicted_bg, data_range=255)
    return background_ssim


def evaluate_single_reasonseg_image(input_path: str, mask_path: str, predicted_path: str, 
                                   color_threshold: int = 50, 
                                   extract_mode: str = "fallback",
                                   color_sensitivity: str = "medium",
                                   g_over_r: float = 1.15, g_over_b: float = 1.15, delta_g: int = 10,
                                   expand_method: str = "regiongrow", lab_sigma: float = 6.0,
                                   min_region_ratio: float = 0.03) -> Optional[Dict]:
    input_img = load_image(input_path)
    predicted_img = load_image(predicted_path)
    gt_mask = load_mask(mask_path)
    target_shape = (512, 512)
    input_img = resize_to_target(input_img, target_shape, is_mask=False)
    predicted_img = resize_to_target(predicted_img, target_shape, is_mask=False)
    gt_mask = resize_to_target(gt_mask, target_shape, is_mask=True)
    edit_mask, method_info = extract_edit_region(input_img, predicted_img, 
                                               color_threshold=color_threshold,
                                               mode=extract_mode,
                                               color_sensitivity=color_sensitivity,
                                               g_over_r=g_over_r,
                                               g_over_b=g_over_b,
                                               delta_g=delta_g,
                                               expand_method=expand_method,
                                               lab_sigma=lab_sigma,
                                               min_region_ratio=min_region_ratio,
                                               return_method_info=True)
    target_value = 255
    iou = calculate_iou(edit_mask, gt_mask, target_value)
    precision, recall = calculate_precision_recall(edit_mask, gt_mask, target_value)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    background_ssim = calculate_simple_background_ssim(input_img, predicted_img, gt_mask)
    results = {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'color_threshold': color_threshold,
        'extract_mode': extract_mode,
        'color_sensitivity': color_sensitivity,
        'background_ssim': float(background_ssim),
        'min_region_ratio': min_region_ratio,
        'method_info': method_info
    }
    return results





def evaluate_model_reasonseg(model_name: str, dataset_dir: str, model_dir: str, 
                            output_dir: str, color_threshold: int = 50, 
                            extract_mode: str = "fallback",
                            color_sensitivity: str = "medium", g_over_r: float = 1.15,
                            g_over_b: float = 1.15, delta_g: int = 10,
                            expand_method: str = "regiongrow", lab_sigma: float = 6.0,
                            min_region_ratio: float = 0.03) -> Dict:
    """
    Evaluate single model performance on ReasonSeg task

    Args:
        model_name: Model name
        dataset_dir: Dataset directory
        model_dir: Model output directory
        output_dir: Output directory
        color_threshold: Green detection tolerance (default: 50)
        extract_mode: Mask extraction mode
        color_sensitivity: Green detection sensitivity
        g_over_r: Minimum ratio of green channel to red channel
        g_over_b: Minimum ratio of green channel to blue channel
        delta_g: Minimum threshold for green channel increment
        expand_method: Expansion method for channel ratio detection
        lab_sigma: Region Growing color threshold
        min_region_ratio: Minimum region area ratio (default: 0.03)

    Returns:
        task_results: Task evaluation result
    """



    file_mappings = get_reasonseg_file_mapping(dataset_dir, model_dir)

    if not file_mappings:
        print(f"Warning: Could not find {model_name} valid files for ReasonSeg task")
        return {
            'task': 'ReasoningPerception',
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


    fallback_ratio_used_count = 0
    fallback_color_only_count = 0
    total_fallback_cases = 0

    for image_id, input_path, mask_path, predicted_path in tqdm(file_mappings, 
                                                               desc="Evaluating ReasonSeg"):
        result = evaluate_single_reasonseg_image(input_path, mask_path, predicted_path, 
                                               color_threshold, extract_mode, color_sensitivity,
                                               g_over_r, g_over_b, delta_g, expand_method, lab_sigma,
                                               min_region_ratio)

        if result is not None:
            result['image_id'] = image_id
            all_results.append(result)
            successful_count += 1

            if extract_mode == 'fallback':
                total_fallback_cases += 1
                method_info = result.get('method_info', {})
                if method_info.get('ratio_used', False):
                    fallback_ratio_used_count += 1
                else:
                    fallback_color_only_count += 1




        else:
            failed_count += 1


    if all_results:

        metrics = ['iou', 'precision', 'recall', 'f1_score', 'background_ssim']
        average_metrics = {}

        for metric in metrics:
            values = [result[metric] for result in all_results if metric in result]
            if values:
                average_metrics[f'avg_{metric}'] = float(np.mean(values))
    else:
        average_metrics = {}


    task_results = {
        'task': 'ReasoningPerception',
        'model': model_name,
        'total_images': len(file_mappings),
        'successful_evaluations': successful_count,
        'failed_evaluations': failed_count,
        'success_rate': successful_count / len(file_mappings) if file_mappings else 0.0,
        'average_metrics': average_metrics,
        'color_threshold': color_threshold,

        'color_sensitivity': color_sensitivity,
        'min_region_ratio': min_region_ratio,
        'fallback_stats': {
            'fallback_ratio_used_count': fallback_ratio_used_count,
            'fallback_color_only_count': fallback_color_only_count,
            'total_fallback_cases': total_fallback_cases
        }
    }

    task_summary_path = os.path.join(model_output_dir, "reasoningperception.json")
    write_json(task_summary_path, task_results)



    return task_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch evaluate model performance on multiple tasks')
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
    parser.add_argument('--color_threshold', 
                       type=int, 
                       default=100, 
                       help='Green detection tolerance range (default: 100)')

    parser.add_argument('--extract_mode', default='fallback', help='Fixed to fallback mode')
    parser.add_argument('--color_sensitivity', 
                       choices=['low', 'medium', 'high'],
                       default='medium',
                       help='Green detection sensitivity: low/medium/high (default: medium); high for very transparent green masks')
    parser.add_argument('--g_over_r', 
                       type=float, 
                       default=1.15, 
                       help='Minimum ratio of green channel to red channel (default: 1.15)')
    parser.add_argument('--g_over_b', 
                       type=float, 
                       default=1.15, 
                       help='Minimum ratio of green channel to blue channel (default: 1.15)')
    parser.add_argument('--delta_g', 
                       type=int, 
                       default=10, 
                       help='Minimum threshold for green channel increment (default: 10)')
    parser.add_argument('--expand_method', 
                       choices=['regiongrow', 'grabcut'],
                       default='regiongrow',
                       help='Expansion method for channel ratio detection: regiongrow (region growing), grabcut (GrabCut) (default: regiongrow)')
    parser.add_argument('--lab_sigma', 
                       type=float, 
                       default=6.0, 
                       help='Region Growing color threshold (default: 6.0)')
    parser.add_argument('--min_region_ratio', 
                       type=float, 
                       default=0.03, 
                       help='Minimum region area ratio; regions smaller than this ratio will be ignored (default: 0.03)')

    args = parser.parse_args()


    ensure_dir(args.output_dir)


    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(args.models_dir) 
                 if os.path.isdir(os.path.join(args.models_dir, d))]

    print(f"Models to evaluate: {models}")
    print(f"Evaluation task: ReasoningPerception")
    print(f"Green detection threshold: {args.color_threshold}")
    print(f"Extraction mode: {args.extract_mode}")

    print(f"Green detection sensitivity: {args.color_sensitivity}")
    print(f"Channel ratio parameters: G/R>{args.g_over_r}, G/B>{args.g_over_b}, ΔG>{args.delta_g}")
    print(f"Expansion method: {args.expand_method}")
    print(f"Lab color threshold: {args.lab_sigma}")
    print(f"Minimum region ratio: {args.min_region_ratio}")
    print("-" * 50)


    all_results = {}

    for model_name in models:
        print(f"\nStart evaluating model: {model_name}")
        model_dir = os.path.join(args.models_dir, model_name)

        model_results = {}

        task_result = evaluate_model_reasonseg(
            model_name=model_name,
            dataset_dir=args.dataset_dir,
            model_dir=model_dir,
            output_dir=args.output_dir,
            color_threshold=args.color_threshold,
            extract_mode=args.extract_mode,
            color_sensitivity=args.color_sensitivity,
            g_over_r=args.g_over_r,
            g_over_b=args.g_over_b,
            delta_g=args.delta_g,
            expand_method=args.expand_method,
            lab_sigma=args.lab_sigma,
            min_region_ratio=args.min_region_ratio
        )
        model_results["ReasoningPerception"] = task_result

        all_results[model_name] = model_results


    minified: Dict[str, float] = {}
    for model_name, model_results in all_results.items():
        task_result = next(iter(model_results.values())) if model_results else {}
        avg_iou = (task_result or {}).get('average_metrics', {}).get('avg_iou', 0.0)
        minified[model_name] = float(avg_iou)
    summary_path = os.path.join(args.output_dir, "reasoningperception_summary.json")
    write_json(summary_path, minified)




if __name__ == "__main__":
    main()
