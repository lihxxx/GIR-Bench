import os
import re
import glob
import json
import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm
from utils import ensure_dir, resolve_models, write_json, parse_image_id


def load_dinov3(device: str = None, dinov3_model_path: str = None, dinov3_repo_dir: str = None):
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Initializing DINOv3 model...")
    if dinov3_repo_dir and dinov3_model_path:
        if not os.path.exists(dinov3_repo_dir):
            print(f"DINOv3 local repo path not found: {dinov3_repo_dir}")
        if not os.path.exists(dinov3_model_path):
            print(f"DINOv3 weights path not found: {dinov3_model_path}")
        print(f"  Loading DINOv3 from local repo: {dinov3_repo_dir}")
        print(f"  Using weights: {dinov3_model_path}")
        model = torch.hub.load(dinov3_repo_dir, 'dinov3_vit7b16', source='local', weights=dinov3_model_path)
    elif dinov3_model_path:
        if not os.path.exists(dinov3_model_path):
            print(f"DINOv3 weights path not found: {dinov3_model_path}")
        print("  Loading DINOv3 from online repo with local weights")
        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb14', weights=dinov3_model_path)
    else:
        print("  Loading DINOv3 pretrained from online repo")
        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb14', pretrained=True)
    model = model.to(device)
    model.eval()
    print("DINOv3 initialized")
    return model, device


def preprocess_pil(img: Image.Image):
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)


def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.load()
    return img


def calculate_dinov3_similarity(model, device: str, img1_pil: Image.Image, img2_pil: Image.Image) -> float:
    t1 = preprocess_pil(img1_pil).unsqueeze(0).to(device)
    t2 = preprocess_pil(img2_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        f1 = model(t1)
        f2 = model(t2)
    f1 = F.normalize(f1, p=2, dim=1)
    f2 = F.normalize(f2, p=2, dim=1)
    return float(F.cosine_similarity(f1, f2).item())


def analyze_model_similarity(model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    print(f"Evaluating model {model_name} on similarity (DINOv3) ...")

    gen_root_categories = os.path.join(args.models_dir, model_name)
    categories = ['Zoology', 'Botany', 'Geography']
    orig_root = args.dataset_dir
    if not any(os.path.isdir(os.path.join(orig_root, c)) for c in categories):
        print(f"Warning: reference categories not found under {orig_root}")
        return {
            'task': 'similarity',
            'model': model_name,
            'metadata': {},
            'category_stats': {},
        }
    print(f"Reference root resolved to: {orig_root}")
    has_any_category = any(os.path.isdir(os.path.join(gen_root_categories, c)) for c in categories)
    if has_any_category:
        gen_root = gen_root_categories
    else:
        print(f"Warning: generated folder not found: {gen_root_categories}/<category>")
        return {
            'task': 'similarity',
            'model': model_name,
            'metadata': {},
            'category_stats': {},
        }

    model, device = load_dinov3(args.device if args.device != 'auto' else None, args.dinov3_model_path, args.dinov3_repo_dir)


    categories = ['Zoology', 'Botany', 'Geography']
    gen_images = []
    for category in categories:
        folder = os.path.join(gen_root, category)
        if os.path.exists(folder):
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp']:
                gen_images.extend(glob.glob(os.path.join(folder, ext)))

    if not gen_images:
        print(f"Warning: no generated images found under {gen_root}")
        return {
            'task': 'similarity',
            'model': model_name,
            'metadata': {},
            'category_stats': {},
        }

    model_output_dir = os.path.join(args.output_dir, model_name)
    ensure_dir(model_output_dir)

    per_image_results: Dict[str, Any] = {}
    category_stats: Dict[str, Any] = {k: {"count": 0, "dinov3": []} for k in ['Zoology', 'Botany', 'Geography']}

    for gen_image_path in tqdm(sorted(gen_images), desc="Analyzing similarity"):
        gen_filename = os.path.basename(gen_image_path)

        category = os.path.basename(os.path.dirname(gen_image_path))
        name_wo_ext = os.path.splitext(gen_filename)[0]


        image_id = parse_image_id(name_wo_ext)
        if image_id is None:
            print(f"Skip {gen_filename}: unexpected name format")
            continue

        if category not in ['Zoology', 'Botany', 'Geography']:
            print(f"Skip {gen_filename}: unsupported category '{category}'")
            continue

        search_categories = [category]

        original_subfolder = None
        for c in search_categories:
            candidates = glob.glob(os.path.join(orig_root, c, f"{image_id}_*"))
            if candidates:
                original_subfolder = candidates[0]
                break
        if not original_subfolder:
            print(f"Skip {gen_filename}: original folder not found for id {image_id}")
            continue

        original_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp']:
            original_images.extend(glob.glob(os.path.join(original_subfolder, ext)))
        if not original_images:
            print(f"Skip {gen_filename}: empty original folder {original_subfolder}")
            continue

        gen_pil = load_and_preprocess_image(gen_image_path)
        if gen_pil is None:
            continue
        dinov3_vals: List[float] = []
        for orig_path in original_images:
            orig_pil = load_and_preprocess_image(orig_path)
            if orig_pil is None:
                continue
            sim = calculate_dinov3_similarity(model, device, gen_pil, orig_pil)
            if sim is not None:
                dinov3_vals.append(sim)

        mapped_category = category
        if mapped_category not in ['Zoology', 'Botany', 'Geography']:
            continue

        per_image_results[gen_filename] = {
            'category': mapped_category,
            'id': image_id,
            'original_subfolder': original_subfolder,
            'num_original_images': len(original_images),
            'dinov3_mean': float(np.mean(dinov3_vals)) if dinov3_vals else None,
            'dinov3_std': float(np.std(dinov3_vals)) if len(dinov3_vals) > 1 else (0.0 if dinov3_vals else None),
            'dinov3_min': float(np.min(dinov3_vals)) if dinov3_vals else None,
            'dinov3_max': float(np.max(dinov3_vals)) if dinov3_vals else None,
        }

        category_stats[mapped_category]['count'] += 1
        if dinov3_vals:
            category_stats[mapped_category]['dinov3'].append(float(np.mean(dinov3_vals)))


    cat_out: Dict[str, Any] = {}
    for cat in ['Zoology', 'Botany', 'Geography']:
        vals = category_stats[cat]['dinov3']
        cat_out[cat] = {
            'count': category_stats[cat]['count'],
            'metrics': {
                'dinov3': {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std': float(np.std(vals)) if len(vals) > 1 else (0.0 if vals else None),
                    'min': float(np.min(vals)) if vals else None,
                    'max': float(np.max(vals)) if vals else None,
                }
            }
        }


    summary = {
        'metadata': {
            'gen_root': gen_root,
            'orig_root': orig_root,
            'total_gen_images': len(per_image_results),
            'metrics': ['dinov3'],
        },
        'category_stats': cat_out,
    }
    write_json(os.path.join(model_output_dir, 'similarity.json'), summary)


    model_summary = {
        'task': 'similarity',
        'model': model_name,
        'summary': summary,
    }

    print(f"Completed model {model_name} - similarity: {summary['metadata']['total_gen_images']} images analyzed")
    return model_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch evaluate models on Text-to-Image similarity (DINOv3)')
    parser.add_argument('--dataset_dir', default='', help='Dataset root directory')
    parser.add_argument('--models_dir', default='', help='Model outputs root directory (images)')
    parser.add_argument('--output_dir', default='', help='Evaluation outputs directory')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate (default: all subdirectories in models_dir)')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--dinov3_model_path', type=str, default=None, help='Path to DINOv3 weights')
    parser.add_argument('--dinov3_repo_dir', type=str, default=None, help='Path to local DINOv3 repo')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, d))]
    print(f"Models to evaluate: {models}")
    print("Task: similarity (DINOv3)")
    print("-" * 50)

    all_results: Dict[str, Any] = {}
    for model_name in models:
        summary = analyze_model_similarity(model_name, args)
        all_results[model_name] = summary

    minified: Dict[str, Dict[str, float]] = {}
    for model_name, summary in all_results.items():
        cats = summary.get('summary', {}).get('category_stats', {}) if isinstance(summary, dict) else {}
        model_cat_means: Dict[str, float] = {}
        for cat in ['Zoology', 'Botany', 'Geography']:
            v = (((cats.get(cat, {}) or {}).get('metrics', {}) or {}).get('dinov3', {}) or {}).get('mean')
            if v is not None:
                model_cat_means[cat] = float(v)
        minified[model_name] = model_cat_means

    summary_json = os.path.join(args.output_dir, 'similarity_summary.json')
    write_json(summary_json, minified)


if __name__ == '__main__':
    main()


