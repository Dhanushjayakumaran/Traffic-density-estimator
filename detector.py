"""
detector.py - Vehicle Detection and Traffic Density Estimation Engine
Uses Ultralytics YOLOv8 and OpenCV to detect and classify road vehicles.
"""

from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
from PIL import Image
import cv2

# COCO dataset class IDs for target vehicles:
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Distinct RGB color palette for annotations (Cinematic surveillance colors)
CLASS_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 229, 255),        # Electric Cyan
    "motorcycle": (255, 170, 0), # Amber Orange
    "bus": (217, 70, 239),       # Neon Magenta / Purple
    "truck": (34, 197, 94),      # Tactical Emerald
}

# Convert RGB to BGR for OpenCV rendering
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    cls: (c[2], c[1], c[0]) for cls, c in CLASS_COLORS_RGB.items()
}


class ModelLoadError(Exception):
    """Custom exception raised when YOLO model loading fails."""
    pass


def load_model(model_name: str = "yolov8n.pt"):
    """
    Load a pretrained YOLO model using Ultralytics.
    
    Args:
        model_name: Model weight identifier or path (default: 'yolov8n.pt').
        
    Returns:
        Loaded YOLO model instance.
        
    Raises:
        ModelLoadError: If Ultralytics is missing or weights cannot be downloaded/loaded.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. "
            "Please install it using: pip install ultralytics"
        ) from e

    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        error_msg = (
            f"Failed to load YOLO model '{model_name}'.\n"
            f"Reason: {str(e)}\n\n"
            "Possible causes and solutions:\n"
            "1. No Internet Access: Ultralytics attempts to auto-download 'yolov8n.pt' on first run.\n"
            "   Ensure your internet connection is active, or manually download 'yolov8n.pt' from:\n"
            "   https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt\n"
            "   and place it in your project folder.\n"
            "2. Corrupted weights file: Delete any existing 'yolov8n.pt' and re-run."
        )
        raise ModelLoadError(error_msg) from e


def estimate_density(
    total_vehicles: int,
    low_threshold: int = 5,
    medium_threshold: int = 15
) -> Dict[str, str]:
    """
    Estimate traffic density based on configurable vehicle count thresholds.
    
    Args:
        total_vehicles: Total number of detected vehicles.
        low_threshold: Maximum vehicle count for 'Low' density.
        medium_threshold: Maximum vehicle count for 'Medium' density.
        
    Returns:
        Dictionary with density level, advisory status, hex color, and description.
    """
    if total_vehicles <= low_threshold:
        return {
            "level": "Low",
            "status": "Free Flowing",
            "color": "#10B981",       # Emerald Green
            "badge_bg": "rgba(16, 185, 129, 0.15)",
            "border_color": "#059669",
            "description": "Traffic is light with smooth vehicle movement. No noticeable congestion or delays.",
        }
    elif total_vehicles <= medium_threshold:
        return {
            "level": "Medium",
            "status": "Moderate Congestion",
            "color": "#F59E0B",       # Amber / Warning
            "badge_bg": "rgba(245, 158, 11, 0.15)",
            "border_color": "#D97706",
            "description": "Moderate vehicle volume detected. Traffic is active with steady flow; minor delays possible.",
        }
    else:
        return {
            "level": "High",
            "status": "Heavy Traffic",
            "color": "#EF4444",       # Crimson Red
            "badge_bg": "rgba(239, 68, 68, 0.15)",
            "border_color": "#DC2626",
            "description": "High vehicle density detected. Significant congestion present; consider alternate routes or signal timing adjustment.",
        }


def draw_annotations(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    mode: str = "all"
) -> np.ndarray:
    """
    Draw sleek, modern surveillance-grade bounding boxes and labels using OpenCV.
    
    Args:
        image_bgr: BGR numpy image array.
        detections: List of detection dictionaries.
        mode: 'all' (boxes + labels), 'boxes_only', 'labels_only', or 'raw'.
        
    Returns:
        Annotated BGR numpy image array.
    """
    if mode == "raw":
        return image_bgr.copy()

    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    # Calculate dynamic line thickness and font scale based on image resolution
    base_dim = max(h, w)
    line_thickness = max(2, int(base_dim / 450))
    font_scale = max(0.42, base_dim / 1600)
    font_thickness = max(1, int(line_thickness / 2))

    for det in detections:
        box = det["bbox"]
        x1, y1, x2, y2 = box
        label = det["class_name"]
        score = det["confidence"]
        color = CLASS_COLORS_BGR.get(label, (255, 255, 255))

        # Draw bounding boxes if requested
        if mode in ("all", "boxes_only"):
            # Primary bounding box with crisp border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

            # High-tech corner HUD brackets
            corner_len = min(int((x2 - x1) * 0.22), int((y2 - y1) * 0.22), 22)
            if corner_len > 4:
                accent_thickness = line_thickness + 2
                # Top-left
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, accent_thickness, cv2.LINE_AA)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, accent_thickness, cv2.LINE_AA)
                # Top-right
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, accent_thickness, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, accent_thickness, cv2.LINE_AA)
                # Bottom-left
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, accent_thickness, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, accent_thickness, cv2.LINE_AA)
                # Bottom-right
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, accent_thickness, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, accent_thickness, cv2.LINE_AA)

        # Draw labels if requested
        if mode in ("all", "labels_only"):
            text = f"{label.upper()} [{int(score * 100)}%]"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Position label above bounding box if space allows; otherwise inside top
            pad = 4
            if y1 - text_h - baseline - (pad * 2) > 0:
                label_y1 = y1 - text_h - baseline - (pad * 2)
                label_y2 = y1
                text_baseline_y = y1 - pad - baseline
            else:
                label_y1 = y1
                label_y2 = y1 + text_h + baseline + (pad * 2)
                text_baseline_y = y1 + text_h + pad

            label_x1 = x1
            label_x2 = min(w, x1 + text_w + (pad * 2))

            # Filled badge background for label
            cv2.rectangle(
                annotated,
                (label_x1, label_y1),
                (label_x2, label_y2),
                color,
                -1
            )

            # Text: Dark surveillance navy for maximum contrast on bright badge
            cv2.putText(
                annotated,
                text,
                (label_x1 + pad, text_baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (6, 10, 20),
                font_thickness,
                cv2.LINE_AA
            )

    return annotated


def generate_spatial_heatmap(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Generate an authentic 2D spatial vehicle density heatmap from actual detections.
    Uses Gaussian kernel density estimation centered on detected vehicle bounding boxes.
    
    Args:
        image_bgr: Source BGR image numpy array.
        detections: List of detection dictionaries.
        
    Returns:
        Annotated BGR image with spatial congestion heatmap overlay.
    """
    h, w = image_bgr.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)

    if not detections:
        # If no vehicles, return darkened image with surveillance grid and clean status
        output = cv2.addWeighted(image_bgr, 0.35, np.zeros_like(image_bgr), 0.65, 0)
        # Draw surveillance grid lines
        for y in range(0, h, max(40, h // 10)):
            cv2.line(output, (0, y), (w, y), (40, 55, 75), 1)
        for x in range(0, w, max(40, w // 10)):
            cv2.line(output, (x, 0), (x, h), (40, 55, 75), 1)
        cv2.putText(
            output,
            "SPATIAL HEATMAP: NO DETECTED VEHICLE HOTSPOTS",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 160, 220),
            2,
            cv2.LINE_AA
        )
        return output

    # Accumulate Gaussian density at each vehicle's spatial footprint
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        box_w = max(10, x2 - x1)
        box_h = max(10, y2 - y1)
        radius = max(25, int(max(box_w, box_h) * 0.75))

        # Stamp a weighted circle on the density map
        cv2.circle(density_map, (cx, cy), radius, 1.0, -1)

    # Blur density map with a large Gaussian kernel for smooth spatial transition
    ksize = int(max(h, w) * 0.15)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(31, min(ksize, 151))
    density_blurred = cv2.GaussianBlur(density_map, (ksize, ksize), 0)

    # Normalize to 0-255 uint8
    max_val = np.max(density_blurred)
    if max_val > 0:
        density_norm = (density_blurred / max_val * 255.0).astype(np.uint8)
    else:
        density_norm = np.zeros((h, w), dtype=np.uint8)

    # Apply TURBO colormap
    heatmap_colored = cv2.applyColorMap(density_norm, cv2.COLORMAP_TURBO)

    # Mask: only show heatmap where density is noticeable
    alpha_mask = cv2.threshold(density_norm, 15, 255, cv2.THRESH_BINARY)[1]
    alpha_mask_3c = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR) / 255.0

    # Dim original image for control room high-contrast presentation
    darkened_bg = (image_bgr.astype(np.float32) * 0.4).astype(np.uint8)

    # Blend: darkened background with heatmap
    blended = (
        darkened_bg * (1.0 - alpha_mask_3c * 0.65) +
        heatmap_colored * (alpha_mask_3c * 0.65)
    ).astype(np.uint8)

    # Overlay subtle tactical grid and coordinate markings
    step_y = max(50, h // 8)
    step_x = max(50, w // 8)
    for y in range(step_y, h, step_y):
        cv2.line(blended, (0, y), (w, y), (35, 50, 70), 1, cv2.LINE_AA)
    for x in range(step_x, w, step_x):
        cv2.line(blended, (x, 0), (x, h), (35, 50, 70), 1, cv2.LINE_AA)

    # Heatmap legend watermark
    cv2.putText(
        blended,
        "SPATIAL VEHICLE CLUSTERING // KERNEL DENSITY ESTIMATION",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 229, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        blended,
        "LOW CONCENTRATION ---------> HIGH CONGESTION HOTSPOT",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 200, 240),
        1,
        cv2.LINE_AA
    )

    return blended


def compute_congestion_insights(
    detections: List[Dict[str, Any]],
    image_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Compute mathematically grounded behavior and congestion metrics from detections.
    
    Args:
        detections: List of detection dictionaries.
        image_shape: (height, width) of the image.
        
    Returns:
        Dictionary of real computed metrics:
            - road_occupancy_pct: % of image area occupied by vehicles
            - avg_confidence: average confidence % across detections
            - dominant_class: vehicle class with the highest presence
            - dominant_pct: % of total vehicles represented by dominant class
            - avg_spacing_px: average nearest-neighbor distance (px)
            - clustering_status: qualitative description of proximity
            - recommendation: tactical control room advisory text
    """
    h, w = image_shape[:2]
    total_image_area = max(1, h * w)

    if not detections:
        return {
            "road_occupancy_pct": 0.0,
            "avg_confidence": 0.0,
            "dominant_class": "None",
            "dominant_pct": 0.0,
            "avg_spacing_px": None,
            "clustering_status": "No Targets Detected",
            "recommendation": "All lanes clear. Unrestricted traffic flow.",
        }

    # 1. Road Surface Area Occupancy Ratio
    total_box_area = sum((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) for d in detections)
    occupancy_pct = min(100.0, (total_box_area / total_image_area) * 100.0)

    # 2. Average Detection Confidence
    avg_conf = (sum(d["confidence"] for d in detections) / len(detections)) * 100.0

    # 3. Dominant Vehicle Class
    class_counts: Dict[str, int] = {}
    for d in detections:
        c = d["class_name"]
        class_counts[c] = class_counts.get(c, 0) + 1

    dom_class = max(class_counts, key=class_counts.get)
    dom_pct = (class_counts[dom_class] / len(detections)) * 100.0

    # 4. Spatial Proximity / Vehicle Clustering
    centroids = [
        ((d["bbox"][0] + d["bbox"][2]) / 2.0, (d["bbox"][1] + d["bbox"][3]) / 2.0)
        for d in detections
    ]
    avg_box_dim = np.mean([
        max(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1])
        for d in detections
    ])

    if len(centroids) >= 2:
        nearest_dists = []
        for i, c1 in enumerate(centroids):
            dists = [
                np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                for j, c2 in enumerate(centroids)
                if i != j
            ]
            if dists:
                nearest_dists.append(min(dists))
        avg_spacing = float(np.mean(nearest_dists)) if nearest_dists else None

        if avg_spacing is not None and avg_box_dim > 0:
            ratio = avg_spacing / avg_box_dim
            if ratio < 1.25:
                clustering_status = "Bumper-to-Bumper (Tight Clustering)"
            elif ratio < 2.5:
                clustering_status = "Normal Queue Spacing"
            else:
                clustering_status = "Dispersed / Open Spacing"
        else:
            clustering_status = "Moderate"
    else:
        avg_spacing = None
        clustering_status = "Single Target (Isolated)"

    # 5. Surveillance Action Recommendation
    if occupancy_pct > 25.0 or len(detections) > 18:
        rec = "CRITICAL CONGESTION: Queue spillover risk. Recommend signal phase extension or rerouting."
    elif occupancy_pct > 10.0 or len(detections) > 8:
        rec = "MODERATE DENSITY: Steady vehicle queuing detected. Monitor bottleneck discharge rate."
    else:
        rec = "FREE-FLOWING: Ample headways observed. Standard signal cycle adequate."

    return {
        "road_occupancy_pct": round(occupancy_pct, 1),
        "avg_confidence": round(avg_conf, 1),
        "dominant_class": dom_class.capitalize(),
        "dominant_pct": round(dom_pct, 1),
        "avg_spacing_px": round(avg_spacing, 1) if avg_spacing is not None else None,
        "clustering_status": clustering_status,
        "recommendation": rec,
    }


def detect_vehicles(
    image_input: Union[Image.Image, np.ndarray, str],
    model: Any,
    confidence_threshold: float = 0.35,
    low_threshold: int = 5,
    medium_threshold: int = 15,
    annotation_mode: str = "all",
) -> Dict[str, Any]:
    """
    Run YOLO vehicle detection and traffic density estimation on an image.
    
    Args:
        image_input: PIL Image, numpy array (RGB), or file path.
        model: Preloaded Ultralytics YOLO model.
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0).
        low_threshold: Upper limit of vehicles for 'Low' density.
        medium_threshold: Upper limit of vehicles for 'Medium' density.
        annotation_mode: Annotation overlay mode ('all', 'boxes_only', 'labels_only', 'raw').
        
    Returns:
        Dictionary containing:
            - original_image: Original RGB PIL Image
            - annotated_image: Annotated RGB PIL Image
            - heatmap_image: Spatial congestion heatmap RGB PIL Image
            - total_count: Total vehicles detected
            - category_counts: Dict mapping vehicle class to count
            - detections: List of individual detection dicts
            - density: Dict with density level, advisory, color, and description
            - insights: Dict with computed congestion and proximity metrics
    """
    # Normalize input image into PIL and numpy RGB
    if isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
        image_rgb = np.array(pil_image)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image_input.copy()
        pil_image = Image.fromarray(image_rgb)
    elif isinstance(image_input, str):
        pil_image = Image.open(image_input).convert("RGB")
        image_rgb = np.array(pil_image)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Convert RGB to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Run inference filtered to vehicle classes (car=2, motorcycle=3, bus=5, truck=7)
    target_class_ids = list(VEHICLE_CLASSES.keys())
    results = model.predict(
        source=image_rgb,
        conf=confidence_threshold,
        classes=target_class_ids,
        verbose=False
    )

    category_counts: Dict[str, int] = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }
    detections: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        first_result = results[0]
        boxes = first_result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()

                class_name = VEHICLE_CLASSES.get(cls_id, "unknown")
                if class_name in category_counts:
                    category_counts[class_name] += 1

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

    # Sort detections by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    total_count = sum(category_counts.values())

    # Draw OpenCV annotations based on requested mode
    annotated_bgr = draw_annotations(image_bgr, detections, mode=annotation_mode)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # Generate real spatial vehicle-density heatmap
    heatmap_bgr = generate_spatial_heatmap(image_bgr, detections)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Estimate traffic density
    density_info = estimate_density(
        total_vehicles=total_count,
        low_threshold=low_threshold,
        medium_threshold=medium_threshold,
    )

    # Compute real mathematical congestion insights
    insights = compute_congestion_insights(detections, (image_rgb.shape[0], image_rgb.shape[1]))

    return {
        "original_image": pil_image,
        "annotated_image": annotated_pil,
        "heatmap_image": heatmap_pil,
        "total_count": total_count,
        "category_counts": category_counts,
        "detections": detections,
        "density": density_info,
        "insights": insights,
    }
