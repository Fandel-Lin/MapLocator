import os
import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from scipy import ndimage
from scipy.ndimage import label, find_objects

_FAST_SLIC_OK = False
_SAM2_OK = False
try:
    from fast_slic.avx2 import SlicAvx2 
    from fast_slic import Slic 
    _FAST_SLIC_OK = True
except Exception:
    try:
        from fast_slic import Slic 
        SlicAvx2 = None 
        _FAST_SLIC_OK = True
    except Exception:
        Slic = None 
        SlicAvx2 = None 

try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator 
    _SAM2_OK = True
except Exception:
    SAM2AutomaticMaskGenerator = None 


@dataclass
class AutoStats:
    input_image_name: str
    color_variety: float
    largest_closed_boundary_ratio: float
    inner_filtered_color_ratio: float
    boundary_outside_color_distinction: float
    assigned_class: str
    sam_available: int
    slic_available: int


# ============================================================================
# Basic shared utilities
# ============================================================================

def ensure_device(device: str = "cuda") -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {path}")
    return img


def gaussian_filter_torch(image: torch.Tensor, radius: float, device: str) -> torch.Tensor:
    kernel_size = int(2 * round(radius) + 1)
    sigma = max(float(radius), 1e-6)
    x_coord = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    gaussian_1d = torch.exp(-0.5 * (x_coord / sigma) ** 2)
    gaussian_1d /= torch.clamp(gaussian_1d.sum(), min=1e-8)
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d).view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2

    if image.dim() == 2:
        image_in = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image_in = image.unsqueeze(0)
    else:
        image_in = image
    smoothed = F.conv2d(image_in.float(), gaussian_2d, padding=padding)
    return smoothed.squeeze(0).squeeze(0)


def save_segmentation_json(mask_np: np.ndarray, output_json_path: str, image_name_id: str, polygonal: bool = False) -> None:
    contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox = [float(x), float(y), float(x + w), float(y + h)]

    if polygonal:
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        approx_curve = cv2.approxPolyDP(largest_contour, epsilon, True)
        poly_bounds = [[float(p[0][0]), float(p[0][1])] for p in approx_curve]
        area = None
        model_id = None
    else:
        poly_bounds = [
            [float(x), float(y)],
            [float(x + w), float(y)],
            [float(x + w), float(y + h)],
            [float(x), float(y + h)],
        ]
        area = float(w * h)
        model_id = "heuristic_strict_cleaned_box"

    json_data = {
        "doc_id": image_name_id,
        "segments": [
            {
                "poly_bounds": poly_bounds,
                "area": area,
                "bbox": bbox,
                "class_label": "map",
                "confidence": 1.0,
                "id_model": model_id,
            }
        ],
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)


def save_outputs(img_bgr: np.ndarray, final_filled: np.ndarray, output_dir: str, name_no_ext: str, polygonal_json: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    name_no_ext = name_no_ext.replace('.cog', '')

    mask_out_path = os.path.join(output_dir, f"{name_no_ext}_mask.tif")
    tifffile.imwrite(mask_out_path, final_filled.astype(np.uint8))

    masked_img = np.full_like(img_bgr, 255)
    mask_bool = final_filled > 0
    masked_img[mask_bool] = img_bgr[mask_bool]
    masked_out_path = os.path.join(output_dir, f"{name_no_ext}_masked.tif")
    tifffile.imwrite(masked_out_path, cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))

    json_out_path = os.path.join(output_dir, f"{name_no_ext}_map_segmentation.json")
    save_segmentation_json(final_filled, json_out_path, name_no_ext, polygonal=polygonal_json)


def _torch_binary_close(bin_mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x = (bin_mask > 127).float().unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    x = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    return (x.squeeze(0).squeeze(0) > 0.5).byte() * 255


def _torch_binary_open(bin_mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x = (bin_mask > 127).float().unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    x = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    return (x.squeeze(0).squeeze(0) > 0.5).byte() * 255


def keep_largest_component(mask_np: np.ndarray) -> np.ndarray:
    labeled, n_comp = label(mask_np > 0)
    if n_comp <= 0:
        return mask_np.astype(np.uint8)
    areas = np.bincount(labeled.ravel())
    areas[0] = 0
    largest_idx = int(np.argmax(areas))
    return ((labeled == largest_idx).astype(np.uint8) * 255)


def fill_holes(mask_np: np.ndarray) -> np.ndarray:
    return (ndimage.binary_fill_holes(mask_np > 0).astype(np.uint8) * 255)


def maybe_write(intermediate_dir: Optional[str], name: str, img: np.ndarray) -> None:
    if intermediate_dir:
        os.makedirs(intermediate_dir, exist_ok=True)
        cv2.imwrite(os.path.join(intermediate_dir, name), img)


def _adaptive_content_mask(img_bgr: np.ndarray, device: str, intermediate_dir: Optional[str], name_no_ext: str,
                           high_trigger: float, high_target: float) -> torch.Tensor:
    h, w, _ = img_bgr.shape
    total_area = float(h * w)

    current_thresh_val = 250

    def make_mask(th: int) -> np.ndarray:
        white_mask = np.all(img_bgr >= th, axis=2)
        return ((~white_mask).astype(np.uint8) * 255)

    content_mask_np = make_mask(current_thresh_val)
    heuristic_coverage = float(np.count_nonzero(content_mask_np) / total_area)
    print(f"  > Initial Heuristic Content Coverage: {heuristic_coverage:.2%}")
    maybe_write(intermediate_dir, f"{name_no_ext}_01_heuristic_init.png", content_mask_np)

    if heuristic_coverage < 0.10:
        print("    [!] Coverage < 10%. Attempting to tolerate lighter pixels...")
        prev_mask = content_mask_np.copy()
        prev_coverage = heuristic_coverage
        for _ in range(6):
            new_thresh_val = min(current_thresh_val + 2, 255)
            if new_thresh_val == current_thresh_val:
                break
            current_thresh_val = new_thresh_val
            content_mask_np = make_mask(current_thresh_val)
            new_coverage = float(np.count_nonzero(content_mask_np) / total_area)
            if new_coverage > 0.20:
                content_mask_np = prev_mask
                heuristic_coverage = prev_coverage
                break
            heuristic_coverage = new_coverage
            prev_mask = content_mask_np.copy()
            prev_coverage = new_coverage
    elif heuristic_coverage > high_trigger:
        print(f"    [!] Coverage > {high_trigger:.0%}. Attempting to be stricter...")
        best_mask_np = content_mask_np.copy()
        best_cov = heuristic_coverage
        for _ in range(12):
            current_thresh_val = max(current_thresh_val - 10, 0)
            candidate = make_mask(current_thresh_val)
            candidate_cov = float(np.count_nonzero(candidate) / total_area)
            if np.count_nonzero(candidate) > 0:
                best_mask_np = candidate
                best_cov = candidate_cov
            content_mask_np = candidate
            heuristic_coverage = candidate_cov
            if heuristic_coverage < high_target:
                if np.count_nonzero(content_mask_np) == 0:
                    content_mask_np = best_mask_np
                    heuristic_coverage = best_cov
                break
        if np.count_nonzero(content_mask_np) == 0:
            content_mask_np = best_mask_np
        content_mask_np = _frame_removal_np(content_mask_np, intermediate_dir, name_no_ext)

    content_mask_np = cv2.GaussianBlur(content_mask_np, (3, 3), 0)
    _, content_mask_np = cv2.threshold(content_mask_np, 127, 255, cv2.THRESH_BINARY)
    for k in (5, 7, 11, 13):
        kernel = np.ones((k, k), np.uint8)
        content_mask_np = cv2.morphologyEx(content_mask_np, cv2.MORPH_CLOSE, kernel)
    maybe_write(intermediate_dir, f"{name_no_ext}_01_heuristic_final.png", content_mask_np)
    return torch.from_numpy(content_mask_np.astype(np.uint8))


def _frame_removal_np(content_mask_cpu: np.ndarray, intermediate_dir: Optional[str], name_no_ext: str) -> np.ndarray:
    lbl_content, n_content = label(content_mask_cpu > 0)
    if n_content <= 0:
        return content_mask_cpu

    h_np, w_np = content_mask_cpu.shape
    lbl_areas = np.bincount(lbl_content.ravel())
    total_px = h_np * w_np
    max_frame_area = total_px * 0.5
    best_frame_id = -1
    final_margin_used = 0
    margin_upper = max(201, min(350, min(h_np, w_np) // 3))

    for check_margin in range(200, margin_upper, 50):
        slices = find_objects(lbl_content)
        candidates = []
        for i, sl in enumerate(slices):
            if sl is None:
                continue
            lbl_id = i + 1
            if lbl_areas[lbl_id] > max_frame_area:
                continue
            dy, dx = sl
            y_start, y_stop = dy.start, dy.stop
            x_start, x_stop = dx.start, dx.stop
            if (y_start < check_margin and y_stop > (h_np - check_margin) and
                x_start < check_margin and x_stop > (w_np - check_margin)):
                candidates.append((lbl_id, int(lbl_areas[lbl_id])))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_frame_id = candidates[0][0]
            final_margin_used = check_margin
            break

    if best_frame_id == -1:
        return content_mask_cpu

    removal_mask_np = np.zeros((h_np, w_np), dtype=np.uint8)
    center_y, center_x = h_np // 2, w_np // 2
    is_connected_to_center = (lbl_content[center_y, center_x] == best_frame_id)
    if not is_connected_to_center:
        removal_mask_np = (lbl_content == best_frame_id).astype(np.uint8) * 255
    else:
        frame_pixels = (lbl_content == best_frame_id).astype(np.uint8) * 255
        masks = []
        mt = np.zeros_like(frame_pixels); mt[:final_margin_used, :] = 255; masks.append(mt)
        mb = np.zeros_like(frame_pixels); mb[h_np-final_margin_used:, :] = 255; masks.append(mb)
        ml = np.zeros_like(frame_pixels); ml[:, :final_margin_used] = 255; masks.append(ml)
        mr = np.zeros_like(frame_pixels); mr[:, w_np-final_margin_used:] = 255; masks.append(mr)
        for zone_mask in masks:
            seg_img = cv2.bitwise_and(frame_pixels, zone_mask)
            if cv2.countNonZero(seg_img) > 0:
                removal_mask_np = cv2.bitwise_or(removal_mask_np, seg_img)

    if cv2.countNonZero(removal_mask_np) <= 0:
        return content_mask_cpu

    frame_mask_dilated = cv2.dilate(removal_mask_np, np.ones((75, 75), np.uint8))
    constraint_mask = np.zeros_like(frame_mask_dilated)
    constraint_mask[:final_margin_used, :] = 255
    constraint_mask[h_np-final_margin_used:, :] = 255
    constraint_mask[:, :final_margin_used] = 255
    constraint_mask[:, w_np-final_margin_used:] = 255
    frame_mask_dilated = cv2.bitwise_and(frame_mask_dilated, constraint_mask)
    maybe_write(intermediate_dir, f"{name_no_ext}_01_frame_exclusion.png", frame_mask_dilated)
    return cv2.bitwise_and(content_mask_cpu, cv2.bitwise_not(frame_mask_dilated))


def _rectify_box(mask_filled: np.ndarray, mode: str, intermediate_dir: Optional[str], name_no_ext: str) -> np.ndarray:
    h, w = mask_filled.shape[:2]
    final_filled = np.zeros_like(mask_filled)

    k_size = max(int(min(h, w) * 0.02), 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    mask_cleaned = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel)
    mask_cleaned = keep_largest_component(mask_cleaned)
    maybe_write(intermediate_dir, f"{name_no_ext}_03a_cleaned_blob.png", mask_cleaned)

    if cv2.countNonZero(mask_cleaned) <= 0:
        maybe_write(intermediate_dir, f"{name_no_ext}_03_rectified_strict.png", final_filled)
        return final_filled

    row_counts = np.count_nonzero(mask_cleaned, axis=1)
    col_counts = np.count_nonzero(mask_cleaned, axis=0)
    non_zero_rows = row_counts[row_counts > 0]
    non_zero_cols = col_counts[col_counts > 0]
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        maybe_write(intermediate_dir, f"{name_no_ext}_03_rectified_strict.png", final_filled)
        return final_filled

    density_factor = 0.60 if mode == "topo" else 0.75
    row_thresh = float(np.median(non_zero_rows) * density_factor)
    col_thresh = float(np.median(non_zero_cols) * density_factor)
    valid_rows = np.where(row_counts > row_thresh)[0]
    valid_cols = np.where(col_counts > col_thresh)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        x_f, y_f, w_f, h_f = cv2.boundingRect(mask_cleaned)
        cv2.rectangle(final_filled, (x_f, y_f), (x_f + w_f, y_f + h_f), 255, -1)
        maybe_write(intermediate_dir, f"{name_no_ext}_03_rectified_strict.png", final_filled)
        return final_filled

    y_start, y_end = int(valid_rows[0]), int(valid_rows[-1])
    x_start, x_end = int(valid_cols[0]), int(valid_cols[-1])

    def adjust_axis(search_slice: np.ndarray, low_from_start: bool, cutoff_mult: float) -> Optional[int]:
        nz = search_slice[search_slice > 0]
        if len(nz) == 0:
            return None
        cutoff = float(np.median(nz) * cutoff_mult)
        idxs = np.where(search_slice <= cutoff)[0]
        if len(idxs) == 0:
            return None
        return int(idxs[-1] if low_from_start else idxs[0])

    top_mult = 0.99 if mode == "topo" else 0.95
    bottom_mult = 0.99 if mode == "topo" else 0.80

    if y_start < 30 or (mode == "topo" and y_start < 300):
        search_limit = int(min((y_end - y_start) * 0.20, 600 if mode == "topo" else 400))
        if search_limit > 20:
            roi = mask_cleaned[y_start:y_start + search_limit, x_start:x_end]
            off = adjust_axis(np.count_nonzero(roi, axis=1), True, top_mult)
            if off is not None:
                new_y_start = y_start + off
                if new_y_start < (y_end - 50):
                    y_start = int(new_y_start)

    img_h = mask_cleaned.shape[0]
    if (mode == "topo" and (y_end > img_h - 30 or y_end > img_h - 300)) or (mode == "pp1300" and y_end > img_h * 0.5):
        search_limit = int(min((y_end - y_start) * (0.40 if mode == "topo" else 0.20), 800 if mode == "topo" else 400))
        if search_limit > 20:
            roi_y1 = max(y_start, y_end - search_limit)
            roi = mask_cleaned[roi_y1:y_end, x_start:x_end]
            off = adjust_axis(np.count_nonzero(roi, axis=1), False, 0.95 if mode == "topo" else bottom_mult)
            if off is not None:
                new_y_end = roi_y1 + off
                if new_y_end > (y_start + 50):
                    y_end = int(new_y_end)

    if mode == "topo":
        img_w = mask_cleaned.shape[1]
        if x_start < 30 or x_start < 300:
            search_limit = int(min((x_end - x_start) * 0.20, 600))
            if search_limit > 20:
                roi = mask_cleaned[y_start:y_end, x_start:x_start + search_limit]
                off = adjust_axis(np.count_nonzero(roi, axis=0), True, 0.95)
                if off is not None:
                    new_x_start = x_start + off
                    if new_x_start < (x_end - 50):
                        x_start = int(new_x_start)
        if x_end > img_w - 30 or x_end > img_w - 300:
            search_limit = int(min((x_end - x_start) * 0.40, 800))
            if search_limit > 20:
                roi_x1 = max(x_start, x_end - search_limit)
                roi = mask_cleaned[y_start:y_end, roi_x1:x_end]
                off = adjust_axis(np.count_nonzero(roi, axis=0), False, 0.95)
                if off is not None:
                    new_x_end = roi_x1 + off
                    if new_x_end > (x_start + 50):
                        x_end = int(new_x_end)

    x_r = int(x_start)
    y_r = int(y_start)
    w_r = max(1, int(x_end - x_start))
    h_r = max(1, int(y_end - y_start))
    cv2.rectangle(final_filled, (x_r, y_r), (x_r + w_r, y_r + h_r), 255, -1)
    maybe_write(intermediate_dir, f"{name_no_ext}_03_rectified_strict.png", final_filled)
    return final_filled


# ============================================================================
# topo / pp1300 logic
# ============================================================================

def execute_topo_or_pp1300_pipeline(img_bgr: np.ndarray, device: str, intermediate_dir: Optional[str], name_no_ext: str,
                                    mode: str = "topo") -> np.ndarray:
    if mode not in {"topo", "pp1300"}:
        raise ValueError(f"Unsupported mode: {mode}")
    print(f"Step 1: Adaptive Heuristic Thresholding ({mode})...")
    high_trigger = 0.50 if mode == "topo" else 0.30
    high_target = 0.50 if mode == "topo" else 0.30
    content_mask = _adaptive_content_mask(img_bgr, device, intermediate_dir, name_no_ext, high_trigger, high_target)

    print("Step 2: Selecting Largest Component and Filling Holes...")
    mask_np = keep_largest_component(content_mask.cpu().numpy().astype(np.uint8))
    mask_filled = fill_holes(mask_np)
    maybe_write(intermediate_dir, f"{name_no_ext}_02_largest_filled.png", mask_filled)

    print("Step 3: Geometric Rectification (Histogram Projection for Strict Box)...")
    final_filled = _rectify_box(mask_filled, mode, intermediate_dir, name_no_ext)
    return final_filled.astype(np.uint8)


def process_topo_or_pp1300(input_path: str, output_dir: str, intermediate_dir: Optional[str], device: str, mode: str) -> np.ndarray:
    filename = os.path.basename(input_path)
    name_no_ext = os.path.splitext(filename)[0]
    print(f"Processing: {filename} [{mode}]")
    img_bgr = read_image_bgr(input_path)
    h, w = img_bgr.shape[:2]
    total_area = float(h * w)

    final_filled = execute_topo_or_pp1300_pipeline(img_bgr, device, intermediate_dir, name_no_ext, mode=mode)
    final_coverage = float(np.count_nonzero(final_filled)) / total_area
    if final_coverage > (0.90 if mode == "pp1300" else 0.88):
        print(f"  > Coverage too high ({final_coverage:.2%}). Initiating forced margin re-process.")
        forced_margin = 100
        extra = 50
        img_bgr_modified = img_bgr.copy()
        img_bgr_modified[:forced_margin + extra, :] = 255
        img_bgr_modified[h-(forced_margin + extra):, :] = 255
        img_bgr_modified[:, :forced_margin + extra] = 255
        img_bgr_modified[:, w-(forced_margin + extra):] = 255
        maybe_write(intermediate_dir, f"{name_no_ext}_99_reprocess_input.png", img_bgr_modified)
        final_filled = execute_topo_or_pp1300_pipeline(img_bgr_modified, device, intermediate_dir, f"{name_no_ext}_reprocess", mode=mode)

    save_outputs(img_bgr, final_filled, output_dir, name_no_ext, polygonal_json=False)
    print(f"Finished {filename}.")
    return final_filled


# ============================================================================
# nickel / SAM-assisted logic
# ============================================================================

def enhance_contrast_and_saturation(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_l = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    l_enhanced = clahe_l.apply(l)
    img_structural = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(img_structural, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe_s = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(8, 8))
    s_enhanced = clahe_s.apply(s)
    return cv2.cvtColor(cv2.merge((h, s_enhanced, v)), cv2.COLOR_HSV2BGR)


def generate_slic_superpixel(image_np: np.ndarray, slic_components: int = 40000, compactness: int = 20) -> np.ndarray:
    if not _FAST_SLIC_OK:
        raise RuntimeError("fast-slic is not available")
    try:
        slic = SlicAvx2(num_components=slic_components, compactness=compactness) if SlicAvx2 is not None else Slic(num_components=slic_components, compactness=compactness)
    except Exception:
        slic = Slic(num_components=slic_components, compactness=compactness)
    assignment = slic.iterate(image_np)
    if assignment.min() < 0:
        if assignment.dtype == np.int16:
            assignment = assignment.view(np.uint16).astype(np.int32)
        else:
            assignment = assignment.astype(np.int32) - int(assignment.min())
    else:
        assignment = assignment.astype(np.int32)
    return assignment


def snap_mask_to_slic(binary_mask_np: np.ndarray, slic_labels_np: np.ndarray) -> np.ndarray:
    if slic_labels_np.min() < 0:
        slic_labels_np = slic_labels_np - slic_labels_np.min()
    flat_slic = slic_labels_np.ravel()
    flat_mask = (binary_mask_np.ravel() > 127).astype(np.int32)
    count = np.bincount(flat_slic)
    sum_mask = np.bincount(flat_slic, weights=flat_mask)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = sum_mask / count
        ratios[count == 0] = 0.0
    keep_labels = np.where(ratios > 0.5)[0]
    return np.isin(slic_labels_np, keep_labels).astype(np.uint8) * 255


def _sam2_masks(input_image_bgr: np.ndarray, target_long_side: int = 1024) -> List[Dict[str, np.ndarray]]:
    if not _SAM2_OK:
        return []
    h, w = input_image_bgr.shape[:2]
    scale = target_long_side / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(input_image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
        "facebook/sam2-hiera-base-plus",
        points_per_side=32,
        points_per_batch=128,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=1000,
        box_nms_thresh=0.7,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = mask_generator.generate(resized)
    # upscale back
    out: List[Dict[str, np.ndarray]] = []
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        seg_full = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append({"segmentation": seg_full, "area": int(seg_full.sum())})
    return out


def _color_mask_nonwhite_nonblack(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    non_white = v < 245
    non_black = v > 20
    colorful = s > 20
    return (non_white & non_black & colorful).astype(np.uint8) * 255


def _largest_closed_boundary_mask(img_bgr: np.ndarray, device: str) -> np.ndarray:
    base = _adaptive_content_mask(img_bgr, device, None, "auto_probe", 0.30, 0.30).cpu().numpy().astype(np.uint8)
    base = keep_largest_component(base)
    base = fill_holes(base)
    return base


def compute_auto_statistics(img_bgr: np.ndarray, input_image_name: str, device: str) -> AutoStats:
    h, w = img_bgr.shape[:2]
    total_area = float(h * w)

    colorful_mask = _color_mask_nonwhite_nonblack(img_bgr)
    largest_closed = _largest_closed_boundary_mask(img_bgr, device)
    largest_closed_bool = largest_closed > 0
    outside_bool = ~largest_closed_bool

    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    valid = ((pixels.sum(axis=1) < 740) & (pixels.sum(axis=1) > 30)).astype(bool)
    valid_pixels = pixels[valid]
    if len(valid_pixels) == 0:
        color_variety = 0.0
    else:
        quant = np.clip((valid_pixels / 32.0).astype(np.int32), 0, 7)
        color_variety = float(np.unique(quant[:, 0] * 64 + quant[:, 1] * 8 + quant[:, 2]).shape[0] / 512.0)

    largest_closed_ratio = float(np.count_nonzero(largest_closed_bool) / total_area)
    inner_filtered_color_ratio = float(np.count_nonzero(colorful_mask[largest_closed_bool]) / max(np.count_nonzero(largest_closed_bool), 1))

    inside_pixels = img_bgr[largest_closed_bool].reshape(-1, 3).astype(np.float32)
    outside_pixels = img_bgr[outside_bool].reshape(-1, 3).astype(np.float32)
    if len(inside_pixels) == 0 or len(outside_pixels) == 0:
        distinction = 0.0
    else:
        in_mean = inside_pixels.mean(axis=0)
        out_mean = outside_pixels.mean(axis=0)
        distinction = float(np.linalg.norm(in_mean - out_mean) / (255.0 * np.sqrt(3)))

    high_a = color_variety >= 0.12
    high_b = largest_closed_ratio >= 0.55
    low_b = largest_closed_ratio < 0.35
    high_c = inner_filtered_color_ratio >= 0.12
    midlow_c = inner_filtered_color_ratio < 0.18
    high_d = distinction >= 0.10
    midlow_d = distinction < 0.12

    if high_a and (high_c or high_d):
        assigned = "nickel"
    elif high_b and (high_c or high_d):
        assigned = "topo"
    elif low_b and midlow_c and midlow_d:
        assigned = "pp1300"
    else:
        assigned = "topo"

    return AutoStats(
        input_image_name=input_image_name,
        color_variety=color_variety,
        largest_closed_boundary_ratio=largest_closed_ratio,
        inner_filtered_color_ratio=inner_filtered_color_ratio,
        boundary_outside_color_distinction=distinction,
        assigned_class=assigned,
        sam_available=int(_SAM2_OK),
        slic_available=int(_FAST_SLIC_OK),
    )


def append_class_statistics(csv_path: str, stats: AutoStats) -> None:
    header = [
        "input_image_name",
        "color_variety",
        "largest_closed_boundary_ratio",
        "inner_filtered_color_ratio",
        "boundary_outside_color_distinction",
        "sam_available",
        "slic_available",
        "assigned_class",
    ]
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            stats.input_image_name,
            f"{stats.color_variety:.6f}",
            f"{stats.largest_closed_boundary_ratio:.6f}",
            f"{stats.inner_filtered_color_ratio:.6f}",
            f"{stats.boundary_outside_color_distinction:.6f}",
            stats.sam_available,
            stats.slic_available,
            stats.assigned_class,
        ])


def execute_nickel_pipeline(img_bgr: np.ndarray, device: str, intermediate_dir: Optional[str], name_no_ext: str,
                            debug_mode: bool = False, poly_smoothness_factor: float = 0.001) -> np.ndarray:
    print("Step 1: Nickel / SAM-assisted pipeline...")
    enhanced = enhance_contrast_and_saturation(img_bgr)
    maybe_write(intermediate_dir, f"{name_no_ext}_01_enhanced.png", enhanced)

    sam_mask = None
    if _SAM2_OK and device == "cuda":
        try:
            print("  > Running SAM2 region proposal...")
            masks = _sam2_masks(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            if masks:
                h, w = img_bgr.shape[:2]
                total_area = float(h * w)
                candidate_union = np.zeros((h, w), dtype=np.uint8)
                kept = 0
                for m in sorted(masks, key=lambda x: x["area"], reverse=True):
                    seg = (m["segmentation"] > 0).astype(np.uint8) * 255
                    if (np.count_nonzero(seg) / total_area) > 0.05:
                        candidate_union = cv2.bitwise_or(candidate_union, seg)
                        kept += 1
                    if kept >= 8:
                        break
                if np.count_nonzero(candidate_union) > 0:
                    sam_mask = candidate_union
                    maybe_write(intermediate_dir, f"{name_no_ext}_02_sam_union.png", sam_mask)
        except Exception as e:
            print(f"  > SAM2 failed, fallback to heuristic only: {e}")
            sam_mask = None

    heuristic = _largest_closed_boundary_mask(img_bgr, device)
    maybe_write(intermediate_dir, f"{name_no_ext}_03_heuristic_seed.png", heuristic)

    fused = heuristic.copy() if sam_mask is None else cv2.bitwise_or(heuristic, sam_mask)
    fused = keep_largest_component(fill_holes(fused))
    fused_t = torch.from_numpy(fused).to(device)
    for k in (9, 13, 17):
        fused_t = _torch_binary_close(fused_t, k)
    fused = fused_t.cpu().numpy().astype(np.uint8)

    if _FAST_SLIC_OK:
        try:
            slic_labels = generate_slic_superpixel(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), slic_components=25000, compactness=16)
            fused = snap_mask_to_slic(fused, slic_labels)
        except Exception as e:
            print(f"  > SLIC snapping failed, continuing without it: {e}")

    fused = keep_largest_component(fill_holes(fused))
    if np.count_nonzero(fused) == 0:
        fused = heuristic

    # polygon smoothing from original nickel spirit
    contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(fused)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        eps = max(1.0, poly_smoothness_factor * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, eps, True)
        cv2.drawContours(out, [approx], -1, 255, thickness=-1)
    else:
        out = fused

    maybe_write(intermediate_dir, f"{name_no_ext}_04_nickel_final.png", out)
    return out.astype(np.uint8)


def classify_map_type_from_stats(
    color_variety: float,
    largest_closed_boundary_ratio: float,
    inner_filtered_color_ratio: float,
    boundary_outside_color_distinction: float,
):
    x = np.array([
        float(color_variety),
        float(largest_closed_boundary_ratio),
        float(inner_filtered_color_ratio),
        float(boundary_outside_color_distinction),
    ], dtype=np.float64)

    x = np.clip(x, 0.0, 1.0)

    mean_ = np.array([
        0.36546694,
        0.55221852,
        0.32427154,
        0.24401230,
    ], dtype=np.float64)

    scale_ = np.array([
        0.21168269,
        0.16735814,
        0.29569304,
        0.18039297,
    ], dtype=np.float64)

    z = (x - mean_) / np.maximum(scale_, 1e-12)
    z1, z2, z3, z4 = z

    q = np.array([
        z1, z2, z3, z4,
        z1*z2, z1*z3, z1*z4, z2*z3, z2*z4, z3*z4,
        z1*z1, z2*z2, z3*z3, z4*z4,
    ], dtype=np.float64)

    W = np.array([
        [-0.05033608, -0.19849238,  1.28817744, -0.99764526,
         -0.63285309,  0.13301948,  0.14591852, -0.01952081, -0.07110305, -0.48645469,
          0.99632571,  0.72227851, -0.36379128,  0.44727186],

        [-1.58127889, -1.99957315, -0.84656304, -0.39260768,
          1.34988341,  0.15218792, -0.06331249,  0.09160022,  1.00813809,  0.90452175,
         -0.28204218,  0.14000367, -0.11728975, -0.50582432],

        [ 1.63161497,  2.19806553, -0.44161440,  1.39025294,
         -0.71703031, -0.28520740, -0.08260603, -0.07207941, -0.93703504, -0.41806706,
         -0.71428353, -0.86228218,  0.48108102,  0.05855246],
    ], dtype=np.float64)

    b = np.array([
        -0.42835389,
        -1.39146129,
         1.81981517,
    ], dtype=np.float64)

    scores = W @ q + b

    classes = ["nickel", "pp1300", "topo"]
    assigned_class = classes[int(np.argmax(scores))]

    class_scores = {
        "nickel": float(scores[0]),
        "pp1300": float(scores[1]),
        "topo": float(scores[2]),
    }

    return assigned_class, class_scores


def classify_map_type_from_stats_with_guardrails(
    color_variety: float,
    largest_closed_boundary_ratio: float,
    inner_filtered_color_ratio: float,
    boundary_outside_color_distinction: float,
):
    a = float(np.clip(color_variety, 0.0, 1.0))
    b = float(np.clip(largest_closed_boundary_ratio, 0.0, 1.0))
    c = float(np.clip(inner_filtered_color_ratio, 0.0, 1.0))
    d = float(np.clip(boundary_outside_color_distinction, 0.0, 1.0))

    # Strong pp1300 signature
    if a <= 0.22 and c <= 0.10 and d <= 0.12:
        return "pp1300", {"nickel": -1e9, "pp1300": 1e9, "topo": -1e9}

    # Strong topo signature
    if b >= 0.70 and d >= 0.22:
        return "topo", {"nickel": -1e9, "pp1300": -1e9, "topo": 1e9}

    # Strong nickel signature
    if a >= 0.58 and (c >= 0.30 or d >= 0.30):
        return "nickel", {"nickel": 1e9, "pp1300": -1e9, "topo": -1e9}

    return classify_map_type_from_stats(a, b, c, d)