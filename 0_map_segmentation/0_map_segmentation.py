import argparse
import os
from datetime import datetime

import numpy as np

from area_segmentor_algorithms import (
    AutoStats,
    classify_map_type_from_stats_with_guardrails,
    append_class_statistics,
    compute_auto_statistics,
    ensure_device,
    execute_nickel_pipeline,
    process_topo_or_pp1300,
    read_image_bgr,
    save_outputs,
)


def process_single_map(
    input_path: str,
    output_dir: str,
    intermediate_dir: str | None,
    device: str = "cuda",
    controller: str = "auto",
    debug_mode: bool = False,
    poly_smoothness_factor: float = 0.001,
    class_statistics_csv: str | None = None,
):
    start_time = datetime.now()
    filename = os.path.basename(input_path)
    name_no_ext = os.path.splitext(filename)[0]
    device = ensure_device(device)

    print(f"[{start_time}] Processing: {filename}")
    print(f"Controller requested: {controller}")
    img_bgr = read_image_bgr(input_path)

    chosen = controller
    stats: AutoStats | None = None
    if controller == "auto":
        stats = compute_auto_statistics(img_bgr, filename, device)

        chosen, class_scores = classify_map_type_from_stats_with_guardrails(
            color_variety=stats.color_variety,
            largest_closed_boundary_ratio=stats.largest_closed_boundary_ratio,
            inner_filtered_color_ratio=stats.inner_filtered_color_ratio,
            boundary_outside_color_distinction=stats.boundary_outside_color_distinction,
        )

        stats.assigned_class = chosen

        print(
            "AUTO statistics | "
            f"color_variety={stats.color_variety:.4f}, "
            f"largest_closed_boundary_ratio={stats.largest_closed_boundary_ratio:.4f}, "
            f"inner_filtered_color_ratio={stats.inner_filtered_color_ratio:.4f}, "
            f"boundary_outside_color_distinction={stats.boundary_outside_color_distinction:.4f}, "
            f"scores={class_scores}, "
            f"assigned={chosen}"
        )

        csv_path = class_statistics_csv or os.path.join(output_dir, "class_statistics.csv")
        append_class_statistics(csv_path, stats)
    elif controller in {"topo", "pp1300", "nickel"}:
        print(f"Using explicit controller: {controller}")
    else:
        raise ValueError("controller must be one of ['topo', 'pp1300', 'nickel', 'auto']")
    #return True

    if chosen in {"topo", "pp1300"}:
        return process_topo_or_pp1300(input_path, output_dir, intermediate_dir, device, chosen)

    if chosen == "nickel":
        final_filled = execute_nickel_pipeline(img_bgr, device, intermediate_dir, name_no_ext, debug_mode, poly_smoothness_factor)
        save_outputs(img_bgr, final_filled, output_dir, name_no_ext, polygonal_json=True)
    else:
        raise RuntimeError(f"Unexpected chosen controller: {chosen}")

    end_time = datetime.now()
    print(f"Finished processing {filename}. Total Processing Time: {end_time - start_time}")
    return final_filled


def main(
    input: str,
    output_dir: str,
    intermediate_dir: str | None,
    device: str = "cuda",
    controller: str = "auto",
    debug_mode: bool = False,
    poly_smoothness_factor: float = 0.001,
    class_statistics_csv: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    if intermediate_dir:
        os.makedirs(intermediate_dir, exist_ok=True)
    return process_single_map(
        input_path=input,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        device=device,
        controller=controller,
        debug_mode=debug_mode,
        poly_smoothness_factor=poly_smoothness_factor,
        class_statistics_csv=class_statistics_csv,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="0_Map_Segmentation")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--intermediate_dir", type=str, nargs="?", default=None, help="Path to intermediate output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument("--controller", type=str, default="auto", choices=["topo", "pp1300", "nickel", "auto"], help="Algorithm controller")
    parser.add_argument("--debug_mode", action="store_true", help="Enable extra debugging images where applicable")
    parser.add_argument("--poly_smoothness_factor", type=float, default=0.001, help="Polygon smoothing factor for nickel")
    parser.add_argument("--class_statistics_csv", type=str, default=None, help="CSV path for auto-classification statistics")
    args = parser.parse_args()
    main(
        input=args.input,
        output_dir=args.output_dir,
        intermediate_dir=args.intermediate_dir,
        device=args.device,
        controller=args.controller,
        debug_mode=args.debug_mode,
        poly_smoothness_factor=args.poly_smoothness_factor,
        class_statistics_csv=args.class_statistics_csv,
    )
