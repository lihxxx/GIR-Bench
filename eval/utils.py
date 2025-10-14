import os
import json
from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import re


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_models(models_dir: str, models: Optional[List[str]]) -> List[str]:
    if models:
        return models
    if not os.path.exists(models_dir):
        return []
    return [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]


def write_json(path: str, data: Any, cls: Any = None, indent: int = 2, ensure_ascii: bool = False) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, cls=cls)


def list_generated_images(model_dir: str,
                          task_subdir: Optional[str] = None,
                          categories: Optional[List[str]] = None,
                          extensions: Optional[List[str]] = None) -> List[str]:
    import glob

    if extensions is None:
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp']

    base = os.path.join(model_dir, task_subdir) if task_subdir else model_dir
    results: List[str] = []

    if categories:
        for cat in categories:
            folder = os.path.join(base, cat)
            if os.path.exists(folder):
                for ext in extensions:
                    results.extend(glob.glob(os.path.join(folder, ext)))
    else:
        if os.path.exists(base):
            for ext in extensions:
                results.extend(glob.glob(os.path.join(base, ext)))

    return sorted(results)


def load_internvl_model(model_path: str,
                        device: str = 'cuda') -> Tuple[Any, Any, None]:
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
        ),
    )
    model.eval()
    return model, processor, None





def denormalize_bboxes(bbox_dict: Dict[str, List[float]],
                       original_hw: Tuple[int, int]) -> Dict[str, List[float]]:
    if not bbox_dict:
        return bbox_dict
    out: Dict[str, List[float]] = {}
    H, W = original_hw
    for key, bbox in bbox_dict.items():
        if not bbox or len(bbox) < 4:
            out[key] = bbox
            continue
        try:
            arr = np.array(bbox).reshape(-1, 4) / 1000.0
            arr[:, 0::2] *= W
            arr[:, 1::2] *= H
            out[key] = arr.flatten().tolist()
        except Exception:
            out[key] = bbox
    return out


def model_inference(model: Any,
                    processor: Any,
                    image: Image.Image,
                    prompt: str,
                    max_new_tokens: int = 1024,
                    do_sample: bool = False) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
        )
        response = processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
    return response



def build_detection_prompt(objects: List[Any]) -> Tuple[List[str], str]:
    if isinstance(objects, list) and all(isinstance(obj, dict) for obj in objects):
        object_descriptions = [
            f"{obj['color']} {obj['name']}" if obj.get('color') else obj['name']
            for obj in objects
        ]
    else:
        object_descriptions = objects

    ref_tags = ','.join([f'<ref>{obj}</ref>' for obj in object_descriptions])
    prompt = f"""<image>
Please provide the bounding box coordinates for the regions described by: {ref_tags}

Return the results in the following format:
{{
    "counts": {{
        "object_name1": count,
        "object_name2": count,
        ...
    }},
    "bboxes": {{
        "object_name1_1": [x1, y1, x2, y2],
        "object_name1_2": [x1, y1, x2, y2],
        "object_name2_1": [x1, y1, x2, y2],
        "object_name2_2": [x1, y1, x2, y2],
        ...
    }},
    "missing_objects": ["missing_object1", "missing_object2", ...]
}}

IMPORTANT: All bounding box coordinates (x1, y1, x2, y2) should be normalized to the range [0, 1000]. 

Please ensure to provide bounding boxes for each object instance that actually exists. If an object type does not exist or cannot be identified, include it in the "missing_objects" list and do not include it in bboxes."""
    return object_descriptions, prompt


def extract_grounding_from_response(text: str, object_names: List[str]) -> Dict[str, Any]:                      
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_text = json_match.group()
        result_data = json.loads(json_text)
        if all(k in result_data for k in ["counts", "bboxes", "missing_objects"]):
            cleaned_bboxes: Dict[str, Any] = {}
            if isinstance(result_data.get('bboxes'), dict):
                for key, value in result_data['bboxes'].items():
                    clean_key = re.sub(r'<[^>]+>', '', key).strip('"')
                    if isinstance(value, str):
                        box_match = re.search(r'<box>\[([^\]]+)\]</box>', value)
                        if box_match:
                            coords_str = box_match.group(1)
                            coords = [int(x.strip()) for x in coords_str.split(',')]
                            if len(coords) >= 4:
                                cleaned_bboxes[clean_key] = coords[:4]
                    elif isinstance(value, list) and len(value) >= 4:
                        cleaned_bboxes[clean_key] = value[:4]
            if cleaned_bboxes:
                result_data['bboxes'] = cleaned_bboxes
            return result_data

    counts = {obj: 0 for obj in object_names}
    bboxes: Dict[str, List[int]] = {}
    missing_objects: List[str] = []
    object_instances: Dict[str, int] = {}

    new_fmt = r'<ref>"?([^"<>]+)"?</ref>\s*:\s*<box>\[([^\]]+)\]</box>'
    matches = re.findall(new_fmt, text)
    if matches:
        for obj_name, coords_str in matches:
            obj_name = obj_name.strip()
            coords = [int(x.strip()) for x in coords_str.split(',')]
            if len(coords) >= 4:
                base_obj_name = obj_name
                if '_' in obj_name:
                    parts = obj_name.rsplit('_', 1)
                    if parts[1].isdigit():
                        base_obj_name = parts[0]
                if base_obj_name in object_names:
                    counts[base_obj_name] += 1
                    bboxes[obj_name] = coords[:4]
    else:                                  
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        bbox_matches = re.findall(bbox_pattern, text)
        if bbox_matches:
            obj_index = 0
            for match in bbox_matches:
                if obj_index < len(object_names):
                    obj_name = object_names[obj_index]
                    x1, y1, x2, y2 = [int(coord) for coord in match]
                    counts[obj_name] += 1
                    if obj_name not in object_instances:
                        object_instances[obj_name] = 0
                    object_instances[obj_name] += 1
                    instance_name = f"{obj_name}_{object_instances[obj_name]}"
                    bboxes[instance_name] = [x1, y1, x2, y2]
                    obj_index += 1
        if not bbox_matches:
            for obj in object_names:
                if obj.lower() in text.lower():
                    counts[obj] = 1
                    object_instances[obj] = 1
                    bboxes[f"{obj}_1"] = []

    for obj in object_names:
        if counts[obj] == 0:
            missing_objects.append(obj)

    return {"counts": counts, "bboxes": bboxes, "missing_objects": missing_objects}


def process_detection_result(result_json: Dict[str, Any],
                             object_names: List[str]) -> Tuple[Dict[str, int], Dict[str, Any], List[str], List[str]]:
    object_counts: Dict[str, int] = {}
    bbox_dict: Dict[str, Any] = {}
    missing_objects: List[str] = []
    excess_objects: List[str] = []

    if "counts" in result_json:
        counts_dict = result_json["counts"]
        for obj in object_names:
            count = counts_dict.get(obj, counts_dict.get(obj.replace(' ', '_'), 0))
            object_counts[obj] = int(count)
        for obj, count in object_counts.items():
            if count == 0 and obj not in missing_objects:
                missing_objects.append(obj)
            elif count > 1:
                excess_objects.append(obj)
    else:
        object_counts = {obj: 0 for obj in object_names}
        missing_objects = list(object_names)

    if "bboxes" in result_json and isinstance(result_json["bboxes"], dict):
        raw = result_json["bboxes"]

        temp_bboxes: Dict[str, Any] = dict(raw)

        base_to_expected: Dict[str, List[str]] = {}
        for exp in object_names:
            parts = exp.split(' ')
            if len(parts) >= 2:
                base = ' '.join(parts[1:])
                base_to_expected.setdefault(base, []).append(exp)


        for key, value in raw.items():
            norm_key = key.replace('_', ' ')

            inst_suffix = ''
            if '_' in key and key.split('_')[-1].isdigit():
                inst_suffix = '_' + key.split('_')[-1]

            if norm_key in object_names:
                bbox_dict[norm_key + inst_suffix] = value
                continue


            base_key = norm_key
            if inst_suffix:
                base_key = norm_key[:-(len(inst_suffix))]
            candidates = base_to_expected.get(base_key, [])
            if candidates:

                target = max(candidates, key=len)
                bbox_dict[target + inst_suffix] = value
            else:

                bbox_dict[key] = value


        base_counts: Dict[str, int] = {}
        for key in bbox_dict.keys():
            if '_' in key and key.split('_')[-1].isdigit():
                base = '_'.join(key.split('_')[:-1]).replace('_', ' ')
            else:
                base = key
            base_counts[base] = base_counts.get(base, 0) + 1
        for obj, cnt in base_counts.items():
            if cnt > 1 and obj not in excess_objects:
                excess_objects.append(obj)

        for obj in object_names:
            bbox_cnt = base_counts.get(obj, 0)
            if obj in object_counts:

                if bbox_cnt > object_counts[obj]:
                    object_counts[obj] = bbox_cnt
                if object_counts[obj] > 0 and obj in missing_objects:
                    missing_objects.remove(obj)
            else:
                object_counts[obj] = bbox_cnt
                if bbox_cnt == 0 and obj not in missing_objects:
                    missing_objects.append(obj)
    else:
        bbox_dict = {}

    if isinstance(result_json.get("missing_objects"), list):
        for obj in result_json["missing_objects"]:
            if obj not in missing_objects:
                missing_objects.append(obj)
        for obj in missing_objects:
            if obj in object_counts:
                object_counts[obj] = 0

    for obj in object_names:
        if obj not in object_counts:
            object_counts[obj] = 0
            if obj not in missing_objects:
                missing_objects.append(obj)

    result_json["excess_objects"] = excess_objects
    return object_counts, bbox_dict, missing_objects, excess_objects




def normalize_id(id_str: str) -> str:
    s = id_str.lstrip('0')
    return s if s != '' else '0'


def parse_image_id(name_without_ext: str, custom_regexes: Optional[List[str]] = None) -> Optional[str]:
    patterns: List[str] = custom_regexes or [r'^(?P<id>\d+)_([A-Za-z]+)$', r'^(?P<id>\d+)$']

    for pat in patterns:
        m = re.match(pat, name_without_ext)
        if not m:
            continue
        if 'id' in m.groupdict():
            return normalize_id(m.group('id'))

        for group in m.groups():
            if group is not None and re.fullmatch(r'\d+', group):
                return normalize_id(group)


    m2 = re.search(r'(\d+)', name_without_ext)
    if m2:
        return normalize_id(m2.group(1))
    return None


