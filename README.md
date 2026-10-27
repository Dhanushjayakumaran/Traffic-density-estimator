# 🚦 Traffic Density Surveillance Console & DSA Analytics Hub

A dense, cinematic, surveillance-grade traffic monitoring web application built with **Streamlit**, **Ultralytics YOLOv8**, **OpenCV**, and **Pillow**, featuring three custom Data Structures and Algorithms (DSA) subsystems:

1. **CO2 | Circular Linked-List Traffic Record Manager**: Sequential cyclic scan auditor maintaining strict `tail.next -> head` pointer invariants.
2. **CO3 | AVL Balanced Binary Search Tree**: Self-balancing BST implementing all four rotation cases (LL, RR, LR, RL) for $O(\log n)$ congestion searching and trend analytics.
3. **CO5 | Road Network Graph & DFS Route Cost Evaluator**: Adjacency list representation with Depth-First Search pathfinding, cycle avoidance, and user-configurable distance vs. congestion weight evaluation.

---

## 📸 Complete System Architecture

### 1. Primary Surveillance View (`📹 LIVE SURVEILLANCE & TELEMETRY`)
- High-resolution road image frame with OpenCV bounding boxes and class tags.
- Integrated upload dropzone (JPG, JPEG, PNG) and `⚡ SCAN IMAGE` trigger.
- Compact Playback-Style Action Controls (`👁️ ALL HUD`, `📦 BOXES ONLY`, `🏷️ LABELS ONLY`, `🔥 HEATMAP`, and `💾 SNAPSHOT`).
- Object Tracking Summary with digital counters for Cars, Trucks, Buses, and Motorcycles.
- 2D Gaussian Kernel Spatial Congestion Heatmap.
- Real Scan History Trends & Mathematical Behavior Insights (road occupancy %, proximity ratio).
- Directional Pollution Sensor Pods (North, South, East, West) labelled `DEMO DATA`.

### 2. CO2 Subsystem (`🔄 CO2 | Circular Linked-List Traffic Records`)
- Implemented via custom Python `CLLNode` and `CircularLinkedList` classes.
- Node schema: `scan_id`, `timestamp`, `car_count`, `truck_count`, `bus_count`, `motorcycle_count`, `total_vehicles`, `congestion_level`, `congestion_score`, `image_filename`.
- Automatically updated on every successful YOLO scan.
- Cyclic interactive navigation (`◀ PREV RECORD` and `NEXT RECORD ▶`).
- Dynamic pointer visualizer (`[Prev: SCAN-A] ➔ [CURRENT: SCAN-B] ➔ [Next: SCAN-C]`).
- Safe deletion (`delete_by_id`) with automatic pointer relinking.
- Invariant verification indicator checking `Tail.next == Head`.

### 3. CO3 Subsystem (`🌳 CO3 | AVL Congestion Analytics`)
- Implemented via custom Python `AVLNode` and `AVLTree` classes.
- Strict height balance factor $|h_L - h_R| \le 1$ maintained through all four rotations:
  - **LL Rotation** (Right rotation on unbalance node)
  - **RR Rotation** (Left rotation on unbalance node)
  - **LR Rotation** (Left rotation on child, then Right rotation on node)
  - **RL Rotation** (Right rotation on child, then Left rotation on node)
- In-order traversal producing deterministically sorted congestion records.
- $O(\log n)$ Exact Search by Scan ID and range filter search.
- Historical trend direction analysis (`Increasing`, `Decreasing`, `Stable`) with congestion score progression chart.
- Clear pedagogical note on historical organization vs future forecasting.

### 4. CO5 Subsystem (`🗺️ CO5 | Road Network & Alternative Routes`)
- Implemented via custom Python `RoadGraph` with an Adjacency List (`Dict[str, List[RoadEdge]]`).
- Sample 7-junction demonstration network (`J1_GATEWAY` through `J7_TERMINAL`).
- Depth-First Search (DFS) with backtracking and visited-set cycle prevention.
- User-adjustable composite cost weighting:
  $$\text{Composite Cost} = (w_{\text{dist}} \times \text{Distance}) + (w_{\text{cong}} \times \text{Congestion})$$
- Automatically evaluates and highlights the lowest-cost candidate route alongside the full comparison table.
- Clear pedagogical note on DFS exploration limits and demonstration network topology.

---

## 📁 Project Structure

```
D:\PROJECTS\TrafficDensityEstimator\
│
├── app.py                # Surveillance control-room dashboard & DSA hub
├── detector.py           # YOLOv8 engine, OpenCV annotations & spatial heatmap
├── dsa_structures.py     # Custom Circular Linked List, AVL Tree, & Road Graph
├── test_dsa.py           # Verification & unit tests for CO2, CO3, and CO5
├── test_logic.py         # Unit tests for congestion & behavior formulas
├── requirements.txt      # Pinned dependencies
├── README.md             # Project & architecture documentation
├── run.ps1               # One-click PowerShell launcher
├── run.bat               # Windows batch launcher (double-click to run)
├── yolov8n.pt            # Pretrained YOLOv8 Nano weights (~6.5 MB)
└── .venv\                # Local Python virtual environment
```

---

## 🧪 Verification & Unit Testing

Run the comprehensive test suites verifying all DSA operations and detection logic:

```powershell
cd D:\PROJECTS\TrafficDensityEstimator
.\.venv\Scripts\Activate.ps1

# Run DSA test suite (CO2, CO3, CO5)
python test_dsa.py

# Run detection & density analytics test suite
python test_logic.py
```

---

## 🚀 How to Launch the Application

In your **PowerShell** terminal:

```powershell
cd D:\PROJECTS\TrafficDensityEstimator
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

*(Alternatively, double-click **`run.bat`** in File Explorer).*
The dashboard will open automatically in your browser at `http://localhost:8501`.
