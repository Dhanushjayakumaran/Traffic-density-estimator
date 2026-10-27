"""
test_logic.py - Unit tests for core traffic density estimation logic and insights
"""

import sys
import unittest
import math

def estimate_density(total_vehicles: int, low_threshold: int = 5, medium_threshold: int = 15):
    if total_vehicles <= low_threshold:
        return {"level": "Low", "status": "Free Flowing", "color": "#10B981"}
    elif total_vehicles <= medium_threshold:
        return {"level": "Medium", "status": "Moderate Congestion", "color": "#F59E0B"}
    else:
        return {"level": "High", "status": "Heavy Traffic", "color": "#EF4444"}


def compute_congestion_insights(detections, image_shape):
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

    total_box_area = sum((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) for d in detections)
    occupancy_pct = min(100.0, (total_box_area / total_image_area) * 100.0)
    avg_conf = (sum(d["confidence"] for d in detections) / len(detections)) * 100.0

    class_counts = {}
    for d in detections:
        c = d["class_name"]
        class_counts[c] = class_counts.get(c, 0) + 1

    dom_class = max(class_counts, key=class_counts.get)
    dom_pct = (class_counts[dom_class] / len(detections)) * 100.0

    return {
        "road_occupancy_pct": round(occupancy_pct, 1),
        "avg_confidence": round(avg_conf, 1),
        "dominant_class": dom_class.capitalize(),
        "dominant_pct": round(dom_pct, 1),
    }


class TestTrafficDensityLogic(unittest.TestCase):

    def test_density_zero_vehicles(self):
        res = estimate_density(0, low_threshold=5, medium_threshold=15)
        self.assertEqual(res["level"], "Low")
        self.assertEqual(res["status"], "Free Flowing")

    def test_density_low_boundary(self):
        res = estimate_density(5, low_threshold=5, medium_threshold=15)
        self.assertEqual(res["level"], "Low")

    def test_density_medium_lower_boundary(self):
        res = estimate_density(6, low_threshold=5, medium_threshold=15)
        self.assertEqual(res["level"], "Medium")

    def test_density_medium_upper_boundary(self):
        res = estimate_density(15, low_threshold=5, medium_threshold=15)
        self.assertEqual(res["level"], "Medium")

    def test_density_high_threshold(self):
        res = estimate_density(16, low_threshold=5, medium_threshold=15)
        self.assertEqual(res["level"], "High")
        self.assertEqual(res["status"], "Heavy Traffic")

    def test_congestion_insights_empty(self):
        res = compute_congestion_insights([], (1080, 1920))
        self.assertEqual(res["road_occupancy_pct"], 0.0)
        self.assertEqual(res["dominant_class"], "None")

    def test_congestion_insights_with_detections(self):
        dets = [
            {"class_name": "car", "confidence": 0.90, "bbox": [100, 100, 200, 200]}, # 100x100 = 10000
            {"class_name": "car", "confidence": 0.80, "bbox": [300, 300, 400, 400]}, # 100x100 = 10000
            {"class_name": "truck", "confidence": 0.70, "bbox": [500, 500, 600, 600]},# 100x100 = 10000
        ]
        # Total area = 1000 x 1000 = 1,000,000
        # Box area = 30,000 => 3.0%
        res = compute_congestion_insights(dets, (1000, 1000))
        self.assertEqual(res["road_occupancy_pct"], 3.0)
        self.assertEqual(res["dominant_class"], "Car")
        self.assertEqual(res["dominant_pct"], 66.7)
        self.assertAlmostEqual(res["avg_confidence"], 80.0, places=1)


if __name__ == "__main__":
    unittest.main()
