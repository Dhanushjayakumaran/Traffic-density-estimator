"""
app.py - Traffic Surveillance Control-Room Console & DSA Analytics Hub
A dense, cinematic, surveillance-grade dashboard powered by YOLOv8, OpenCV, and custom DSA components:
- CO2: Circular Linked-List Traffic Record Manager
- CO3: AVL Balanced Binary Search Tree for Congestion Trends & Searching
- CO5: Road Network Graph, Adjacency List, & DFS Alternative Route Cost Evaluator
"""

import io
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import streamlit as st
from PIL import Image
import numpy as np

# Page configuration - Surveillance Dark Theme, Wide Screen, Minimal Margins
st.set_page_config(
    page_title="TRAFFIC SURVEILLANCE CONSOLE // CAM-01",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Surveillance Control Room Styling (Cinematic Dark Navy, Slate, Cyan & Red Accents)
st.markdown(
    """
    <style>
    /* Global Reset & Dark Surveillance Theme */
    html, body, [class*="css"] {
        background-color: #04070F !important;
        color: #D1D5DB !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .stApp {
        background: #04070F !important;
        background-image: 
            radial-gradient(rgba(14, 28, 54, 0.4) 1px, transparent 1px),
            linear-gradient(180deg, #050A14 0%, #03060B 100%) !important;
        background-size: 24px 24px, 100% 100% !important;
    }

    /* Screen-Fit Layout with Minimal Scrolling */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.8rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }

    /* Top Control-Room Bar */
    .surv-topbar {
        background: #08101E;
        border: 1px solid #19263E;
        border-radius: 4px;
        padding: 8px 14px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    .surv-title-main {
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .surv-subtag {
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 0.68rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Live Flashing Red Recording Indicator */
    .rec-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #EF4444;
        box-shadow: 0 0 8px #EF4444;
        animation: pulse-red 1.6s infinite ease-in-out;
    }
    @keyframes pulse-red {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.35; transform: scale(0.85); }
    }

    /* Surveillance Card / Control Panel */
    .ctrl-panel {
        background: #090F1B;
        border: 1px solid #17243B;
        border-radius: 4px;
        padding: 10px 12px;
        margin-bottom: 10px;
        position: relative;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    .ctrl-panel:hover {
        border-color: #23385A;
    }

    /* Control Panel Header */
    .ctrl-header {
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        border-bottom: 1px solid #142034;
        padding-bottom: 5px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .ctrl-header-title {
        color: #38BDF8;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .ctrl-tag {
        background: rgba(56, 189, 248, 0.1);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 1px 6px;
        border-radius: 2px;
        font-size: 0.62rem;
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
    }

    /* Digital Readout / Metric Counters */
    .surv-counter-box {
        background: #060B14;
        border: 1px solid #142033;
        border-radius: 3px;
        padding: 8px 10px;
        text-align: center;
    }
    .surv-counter-val {
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 1.7rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .surv-counter-lbl {
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 0.65rem;
        color: #64748B;
        text-transform: uppercase;
        margin-top: 2px;
        letter-spacing: 0.05em;
    }

    /* Compact Controls & Button Styling */
    div.stButton > button {
        background: #0E1A2E;
        color: #38BDF8;
        border: 1px solid #23395B;
        border-radius: 3px;
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        transition: all 0.15s ease;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div.stButton > button:hover {
        background: #152642;
        border-color: #38BDF8;
        color: #FFFFFF;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }

    /* Scan Image Glow Button */
    .scan-btn-container div.stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        box-shadow: 0 0 14px rgba(2, 132, 199, 0.45) !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        padding: 6px 18px !important;
    }
    .scan-btn-container div.stButton > button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #0284C7 100%) !important;
        border-color: #7DD3FC !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.7) !important;
    }

    /* File Uploader Compact Styling */
    section[data-testid="stFileUploadDropzone"] {
        background: #060B14 !important;
        border: 1px dashed #23395B !important;
        border-radius: 4px !important;
        padding: 8px !important;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #38BDF8 !important;
    }

    /* Primary Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #060B14;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid #142033;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 0.76rem;
        color: #64748B;
        padding: 6px 14px;
        border-radius: 3px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #111D33 !important;
        color: #38BDF8 !important;
        font-weight: 700;
        border: 1px solid #23395B;
    }

    /* Empty state styling */
    .empty-state-box {
        background: #060B14;
        border: 1px dashed #1B2B45;
        border-radius: 4px;
        padding: 24px 16px;
        text-align: center;
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        color: #475569;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
    }

    /* Sensor & Route Telemetry Cards */
    .sensor-card {
        background: #060B14;
        border: 1px solid #142033;
        border-radius: 3px;
        padding: 8px 10px;
        margin-bottom: 6px;
    }
    .route-card {
        background: #081120;
        border: 1px solid #192D4A;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .route-card-best {
        background: #091829;
        border: 1px solid #00E5FF;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.2);
    }

    /* Disclaimer Alert Box */
    .dsa-notice {
        background: #060B14;
        border: 1px solid #1E293B;
        border-left: 3px solid #38BDF8;
        padding: 8px 12px;
        margin-bottom: 10px;
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 0.72rem;
    }

    /* Pointer Visualization Pill */
    .pointer-chain {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #060B14;
        border: 1px solid #17243B;
        padding: 8px 12px;
        border-radius: 4px;
        font-family: 'SF Mono', 'Roboto Mono', 'Consolas', monospace;
        font-size: 0.75rem;
        overflow-x: auto;
    }
    .pointer-node {
        background: #0E1A2E;
        border: 1px solid #23395B;
        padding: 4px 8px;
        border-radius: 3px;
        color: #94A3B8;
        white-space: nowrap;
    }
    .pointer-active {
        background: #0369A1;
        border-color: #38BDF8;
        color: #FFFFFF;
        font-weight: 700;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.35);
    }
    .pointer-arrow {
        color: #38BDF8;
        font-weight: 800;
    }

    /* Sidebar Dark */
    section[data-testid="stSidebar"] {
        background-color: #060B14 !important;
        border-right: 1px solid #142033 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load detection and analytics engine
try:
    from detector import (
        load_model,
        detect_vehicles,
        draw_annotations,
        generate_spatial_heatmap,
        compute_congestion_insights,
        estimate_density,
        ModelLoadError,
        VEHICLE_CLASSES,
        CLASS_COLORS_RGB,
    )
except ImportError as e:
    st.error(f"Failed to import 'detector.py' or missing dependencies. Error: {e}")
    st.stop()

# Load Core DSA Components
try:
    from dsa_structures import (
        TrafficRecord,
        CircularLinkedList,
        AVLTree,
        RoadGraph,
        create_sample_road_network,
    )
except ImportError as e:
    st.error(f"Failed to import 'dsa_structures.py'. Ensure the file is in the project folder. Error: {e}")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_cached_yolo_model(model_name: str = "yolov8n.pt"):
    """Cached loader guaranteeing single in-memory model instantiation."""
    return load_model(model_name)


def init_session_state():
    """Initialize scan history, control state, and DSA structures."""
    if "scan_history" not in st.session_state:
        st.session_state["scan_history"] = []
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "all"
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "analyzed_image_name" not in st.session_state:
        st.session_state["analyzed_image_name"] = None

    # DSA Structure Persistent Singletons
    if "cll_manager" not in st.session_state:
        st.session_state["cll_manager"] = CircularLinkedList()
    if "cll_nav_index" not in st.session_state:
        st.session_state["cll_nav_index"] = 0
    if "avl_tree" not in st.session_state:
        st.session_state["avl_tree"] = AVLTree()
    if "road_graph" not in st.session_state:
        st.session_state["road_graph"] = create_sample_road_network()


def main():
    init_session_state()

    cll: CircularLinkedList = st.session_state["cll_manager"]
    avl: AVLTree = st.session_state["avl_tree"]
    graph: RoadGraph = st.session_state["road_graph"]

    # ----------------------------------------------------
    # TOP SURVEILLANCE TELEMETRY BAR
    # ----------------------------------------------------
    cll_count = len(cll)
    avl_count = len(avl)
    st.markdown(
        f"""
        <div class="surv-topbar">
            <div class="surv-title-main">
                <span class="rec-indicator"></span>
                <span>TRAFFIC SURVEILLANCE CONSOLE</span>
                <span class="ctrl-tag" style="border-color:#10B981; color:#10B981; background:rgba(16,185,129,0.1);">LIVE ARMED</span>
            </div>
            <div style="display:flex; align-items:center; gap:14px;">
                <span class="surv-subtag">CAM-01 // SECTOR 04 HIGHWAY JUNCTION</span>
                <span class="ctrl-tag" style="border-color:#38BDF8;">CLL: {cll_count} RECORDS</span>
                <span class="ctrl-tag" style="border-color:#38BDF8;">AVL: {avl_count} NODES</span>
                <span class="ctrl-tag">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------
    # SIDEBAR: Configuration & Sensitivity Controls
    # ----------------------------------------------------
    with st.sidebar:
        st.markdown("<div class='ctrl-header'><span class='ctrl-header-title'>⚙️ TELEMETRY CONFIG</span></div>", unsafe_allow_html=True)
        
        model_choice = st.selectbox(
            "YOLO Model Weights",
            ["yolov8n.pt (Nano - Real-time)", "yolov8s.pt (Small - High Precision)"],
            index=0,
            help="Nano is optimized for CPU inference. Small offers higher detection recall.",
        )
        model_weight_file = model_choice.split()[0]

        confidence_threshold = st.slider(
            "Detection Confidence Gate",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Minimum confidence threshold required to confirm vehicle targets.",
        )

        st.markdown("<div class='ctrl-header' style='margin-top:16px;'><span class='ctrl-header-title'>🚦 DENSITY LIMITS</span></div>", unsafe_allow_html=True)
        low_limit = st.slider("Low Density Cutoff (≤ vehicles)", 1, 20, 5, 1)
        med_limit = st.slider("Medium Density Cutoff (≤ vehicles)", low_limit + 1, 40, max(low_limit + 1, 15), 1)

        st.markdown("<div class='ctrl-header' style='margin-top:16px;'><span class='ctrl-header-title'>🧩 DSA SUBSYSTEMS</span></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-family:monospace; font-size:0.72rem; color:#94A3B8; line-height:1.6;">
                <div>• <b>CO2:</b> Circular Linked List ({cll_count} stored)</div>
                <div>• <b>CO3:</b> AVL Tree Height ({avl.get_tree_stats()['tree_height']}, Balanced: {avl.get_tree_stats()['is_strictly_balanced']})</div>
                <div>• <b>CO5:</b> Road Graph ({len(graph.get_intersections())} Junctions, 10 Edges)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='ctrl-header' style='margin-top:16px;'><span class='ctrl-header-title'>🎨 TARGET COLOR CODING</span></div>", unsafe_allow_html=True)
        for cls_name, color_rgb in CLASS_COLORS_RGB.items():
            hex_color = "#{:02x}{:02x}{:02x}".format(*color_rgb)
            st.markdown(
                f"<div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:4px; font-family:monospace; font-size:0.75rem;'>"
                f"<span style='color:#CBD5E1;'>{cls_name.upper()}</span>"
                f"<span style='background:{hex_color}; width:16px; height:8px; border-radius:2px;'></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if st.session_state["scan_history"]:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Scan History & DSA"):
                st.session_state["scan_history"] = []
                st.session_state["last_result"] = None
                st.session_state["cll_manager"] = CircularLinkedList()
                st.session_state["avl_tree"] = AVLTree()
                st.session_state["cll_nav_index"] = 0
                st.rerun()

    # Model Loading with Descriptive Diagnostics
    try:
        model = get_cached_yolo_model(model_weight_file)
    except ModelLoadError as mle:
        st.error(f"### ⚠️ YOLO Model Acquisition Error: {mle}")
        st.stop()
    except Exception as e:
        st.error(f"Inference engine failure: {e}")
        st.stop()

    # ----------------------------------------------------
    # PRIMARY NAVIGATION TABS (Preserves surveillance feed and integrates CO2, CO3, CO5)
    # ----------------------------------------------------
    tab_surv, tab_co2, tab_co3, tab_co5 = st.tabs([
        "📹 LIVE SURVEILLANCE & TELEMETRY",
        "🔄 CO2 | Circular Linked-List Traffic Records",
        "🌳 CO3 | AVL Congestion Analytics",
        "🗺️ CO5 | Road Network & Alternative Routes",
    ])

    # ==========================================================================
    # TAB 1: LIVE SURVEILLANCE FEED & TELEMETRY (EXISTING COMPLETE DASHBOARD)
    # ==========================================================================
    with tab_surv:
        col_main_feed, col_side_panels = st.columns([64, 36])

        # ------------------------------------------------
        # UPPER-LEFT MAIN CAMERA PANEL
        # ------------------------------------------------
        with col_main_feed:
            st.markdown(
                """
                <div class="ctrl-panel">
                    <div class="ctrl-header">
                        <span class="ctrl-header-title">
                            <span>📹</span>
                            <span>PRIMARY SURVEILLANCE FEED // CAM-01</span>
                        </span>
                        <div style="display:flex; gap:6px;">
                            <span class="ctrl-tag" style="border-color:#38BDF8;">FEED ACTIVE</span>
                            <span class="ctrl-tag">FPS: 30</span>
                            <span class="ctrl-tag">COCO-VEHICLE GATE</span>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Upload Control & Scan Action directly within/above main panel
            up_col1, up_col2 = st.columns([72, 28])
            with up_col1:
                uploaded_file = st.file_uploader(
                    "Upload Surveillance Frame (JPG, JPEG, PNG)",
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed",
                )
            with up_col2:
                st.markdown("<div class='scan-btn-container'>", unsafe_allow_html=True)
                scan_clicked = st.button("⚡ SCAN IMAGE", key="main_scan_btn", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Sample Scene shortcut if user has no file
            sub_ctrl_1, sub_ctrl_2 = st.columns([60, 40])
            with sub_ctrl_1:
                st.caption("Upload road image frame (JPG, JPEG, PNG) to initiate YOLOv8 telemetry.")
            with sub_ctrl_2:
                use_sample = st.button("🧪 LOAD SAMPLE ROAD", use_container_width=True)

            # Resolve active image
            active_image: Optional[Image.Image] = None
            image_name = ""

            if uploaded_file is not None:
                active_image = Image.open(uploaded_file).convert("RGB")
                image_name = uploaded_file.name
            elif use_sample or st.session_state.get("sample_active", False):
                st.session_state["sample_active"] = True
                active_image = create_synthetic_road_sample()
                image_name = "sample_surveillance_frame.png"

            # Inference Execution
            if active_image is not None:
                needs_scan = scan_clicked or (
                    st.session_state["last_result"] is None and st.session_state.get("auto_scanned") != image_name
                ) or (
                    st.session_state.get("analyzed_image_name") != image_name and scan_clicked
                )

                if needs_scan:
                    with st.spinner("Processing neural inference on surveillance frame..."):
                        result = detect_vehicles(
                            image_input=active_image,
                            model=model,
                            confidence_threshold=confidence_threshold,
                            low_threshold=low_limit,
                            medium_threshold=med_limit,
                            annotation_mode="all",
                        )
                        st.session_state["last_result"] = result
                        st.session_state["analyzed_image_name"] = image_name
                        st.session_state["auto_scanned"] = image_name

                        # Construct unified TrafficRecord for scan history, CLL, and AVL Tree
                        scan_num = len(st.session_state["scan_history"]) + 1
                        scan_id_str = f"SCAN-{scan_num:03d}"
                        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        traffic_rec = TrafficRecord(
                            scan_id=scan_id_str,
                            timestamp=timestamp_now,
                            car_count=result["category_counts"]["car"],
                            truck_count=result["category_counts"]["truck"],
                            bus_count=result["category_counts"]["bus"],
                            motorcycle_count=result["category_counts"]["motorcycle"],
                            total_vehicles=result["total_count"],
                            congestion_level=result["density"]["level"],
                            congestion_score=float(result["total_count"]),
                            image_filename=image_name,
                        )

                        # 1. Update basic history log
                        st.session_state["scan_history"].append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "scan_id": scan_num,
                            "total": result["total_count"],
                            "car": result["category_counts"]["car"],
                            "truck": result["category_counts"]["truck"],
                            "bus": result["category_counts"]["bus"],
                            "motorcycle": result["category_counts"]["motorcycle"],
                            "density": result["density"]["level"],
                        })

                        # 2. Update CO2: Circular Linked-List
                        cll.append(traffic_rec)
                        st.session_state["cll_nav_index"] = len(cll) - 1

                        # 3. Update CO3: AVL Balanced Binary Search Tree
                        avl.insert(traffic_rec)

                # Retrieve active result if available
                result = st.session_state.get("last_result")

                # Determine rendered frame based on active view mode
                active_view = st.session_state.get("view_mode", "all")
                if result is not None:
                    if active_view == "raw":
                        display_img = result["original_image"]
                    elif active_view == "heatmap":
                        display_img = result["heatmap_image"]
                    elif active_view in ("boxes_only", "labels_only"):
                        raw_bgr = cv2.cvtColor(np.array(result["original_image"]), cv2.COLOR_RGB2BGR)
                        ann_bgr = draw_annotations(raw_bgr, result["detections"], mode=active_view)
                        display_img = Image.fromarray(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB))
                    else:
                        display_img = result["annotated_image"]
                else:
                    display_img = active_image

                # Display surveillance monitor feed
                st.image(
                    display_img,
                    caption=f"SURVEILLANCE FRAME // {image_name} ({active_image.width}×{active_image.height} px)",
                    use_container_width=True,
                )

                # Compact Playback-Style Action Controls below the image
                st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
                p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([22, 20, 20, 20, 18])

                with p_col1:
                    if st.button("👁️ ALL HUD", use_container_width=True):
                        st.session_state["view_mode"] = "all"
                        st.rerun()
                with p_col2:
                    if st.button("📦 BOXES ONLY", use_container_width=True):
                        st.session_state["view_mode"] = "boxes_only"
                        st.rerun()
                with p_col3:
                    if st.button("🏷️ LABELS ONLY", use_container_width=True):
                        st.session_state["view_mode"] = "labels_only"
                        st.rerun()
                with p_col4:
                    if st.button("🔥 HEATMAP", use_container_width=True):
                        st.session_state["view_mode"] = "heatmap"
                        st.rerun()
                with p_col5:
                    buf = io.BytesIO()
                    display_img.save(buf, format="PNG")
                    st.download_button(
                        label="💾 SNAPSHOT",
                        data=buf.getvalue(),
                        file_name=f"surveillance_capture_{int(time.time())}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

            else:
                st.markdown(
                    """
                    <div class="empty-state-box" style="padding: 70px 20px;">
                        <div style="font-size: 1.8rem; margin-bottom: 8px;">📡</div>
                        <div style="color: #38BDF8; font-weight: 700; margin-bottom: 4px;">SURVEILLANCE SENSOR STANDBY</div>
                        <div>UPLOAD A ROAD IMAGE OR CLICK 'LOAD SAMPLE ROAD' TO INITIATE LIVE SCAN</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------------------------
        # UPPER-RIGHT PANELS
        # ------------------------------------------------
        with col_side_panels:
            curr_res = st.session_state.get("last_result")
            total_detected = curr_res["total_count"] if curr_res else 0
            cat_counts = curr_res["category_counts"] if curr_res else {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
            density_info = curr_res["density"] if curr_res else {
                "level": "Standby",
                "status": "Awaiting Scan",
                "color": "#64748B",
                "badge_bg": "rgba(100, 116, 139, 0.1)",
                "border_color": "#334155",
                "description": "Upload a frame to compute vehicle volume.",
            }

            # 1. Object Tracking Summary
            st.markdown(
                f"""
                <div class="ctrl-panel">
                    <div class="ctrl-header">
                        <span class="ctrl-header-title">
                            <span>🎯</span>
                            <span>OBJECT TRACKING SUMMARY</span>
                        </span>
                        <span class="ctrl-tag" style="border-color:{density_info['border_color']}; color:{density_info['color']};">
                            {density_info['level'].upper()} DENSITY
                        </span>
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-bottom: 8px;">
                        <div class="surv-counter-box">
                            <div class="surv-counter-val" style="color:#00E5FF;">{cat_counts['car']}</div>
                            <div class="surv-counter-lbl">Cars</div>
                        </div>
                        <div class="surv-counter-box">
                            <div class="surv-counter-val" style="color:#22C55E;">{cat_counts['truck']}</div>
                            <div class="surv-counter-lbl">Trucks</div>
                        </div>
                        <div class="surv-counter-box">
                            <div class="surv-counter-val" style="color:#D946EF;">{cat_counts['bus']}</div>
                            <div class="surv-counter-lbl">Buses</div>
                        </div>
                        <div class="surv-counter-box">
                            <div class="surv-counter-val" style="color:#FFAA00;">{cat_counts['motorcycle']}</div>
                            <div class="surv-counter-lbl">Bikes</div>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; background:#060B14; border:1px solid #142033; padding:6px 10px; border-radius:3px;">
                        <span class="surv-subtag">TOTAL ACTIVE TARGETS</span>
                        <span style="font-family:monospace; font-size:1.15rem; font-weight:800; color:{density_info['color']};">{total_detected}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Spatial Congestion Heatmap
            st.markdown(
                """
                <div class="ctrl-panel">
                    <div class="ctrl-header">
                        <span class="ctrl-header-title">
                            <span>🗺️</span>
                            <span>SPATIAL VEHICLE DENSITY HEATMAP</span>
                        </span>
                        <span class="ctrl-tag">2D KERNEL DENSITY</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if curr_res is not None and curr_res.get("heatmap_image") is not None:
                st.image(
                    curr_res["heatmap_image"],
                    caption="Spatial vehicle clustering computed from detected bounding centroids",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    """
                    <div class="empty-state-box" style="padding: 24px 10px;">
                        <div style="font-size:1.2rem; margin-bottom:4px;">🧭</div>
                        <div>AWAITING SCAN FOR SPATIAL DENSITY COMPUTATION</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # 3. Traffic Statistics & Trends (Real Scan History Only)
            st.markdown(
                """
                <div class="ctrl-panel">
                    <div class="ctrl-header">
                        <span class="ctrl-header-title">
                            <span>📈</span>
                            <span>TRAFFIC STATISTICS & TRENDS</span>
                        </span>
                        <span class="ctrl-tag">HISTORICAL TELEMETRY</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            history = st.session_state["scan_history"]
            if len(history) >= 2:
                trend_data = {
                    "Scan": [f"#{item['scan_id']} ({item['timestamp']})" for item in history],
                    "Vehicles": [item["total"] for item in history],
                }
                st.line_chart(trend_data, x="Scan", y="Vehicles", height=110)
                delta = history[-1]["total"] - history[-2]["total"]
                delta_color = "#EF4444" if delta > 0 else ("#10B981" if delta < 0 else "#38BDF8")
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; font-family:monospace; font-size:0.7rem;'>"
                    f"<span style='color:#64748B;'>SCANS RECORDED: {len(history)}</span>"
                    f"<span style='color:{delta_color}; font-weight:700;'>DELTA: {delta:+d} vs previous</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="empty-state-box">
                        <div style="color: #64748B; margin-bottom: 2px;">[WAITING FOR SCANS // INSUFFICIENT TEMPORAL DATA]</div>
                        <div style="color: #475569; font-size: 0.65rem;">Perform multiple scans to record volume trend telemetry.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------------------------
        # LOWER 3 PANELS: DETECTION STATS, BEHAVIOR, DISTRIBUTION
        # ------------------------------------------------
        low_col1, low_col2, low_col3 = st.columns(3)

        with low_col1:
            st.markdown(
                """
                <div class="ctrl-panel">
                    <div class="ctrl-header">
                        <span class="ctrl-header-title"><span>📊</span><span>DETECTION STATISTICS</span></span>
                        <span class="ctrl-tag">CLASSIFICATION</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if curr_res is not None and curr_res["detections"]:
                insights = curr_res.get("insights", {})
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-family:monospace; font-size:0.75rem;">
                        <span style="color:#94A3B8;">Total Targets Confirmed:</span>
                        <span style="color:#FFFFFF; font-weight:700;">{curr_res['total_count']}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-family:monospace; font-size:0.75rem;">
                        <span style="color:#94A3B8;">Mean Confidence:</span>
                        <span style="color:#38BDF8; font-weight:700;">{insights.get('avg_confidence', 0)}%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-family:monospace; font-size:0.75rem;">
                        <span style="color:#94A3B8;">Dominant Category:</span>
                        <span style="color:#FFAA00; font-weight:700;">{insights.get('dominant_class', 'N/A')} ({insights.get('dominant_pct', 0)}%)</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-family:monospace; font-size:0.75rem;">
                        <span style="color:#94A3B8;">Density Status:</span>
                        <span style="color:{curr_res['density']['color']}; font-weight:700;">{curr_res['density']['level'].upper()} ({curr_res['density']['status']})</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='empty-state-box'><div>[NO ACTIVE TARGETS DETECTED]</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with low_col2:
            st.markdown(
                """
                <div class="ctrl-panel">
                    <div class="ctrl-header">
                        <span class="ctrl-header-title"><span>🧠</span><span>BEHAVIOR & CONGESTION INSIGHTS</span></span>
                        <span class="ctrl-tag">MATHEMATICAL METRICS</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if curr_res is not None and curr_res["detections"]:
                insights = curr_res.get("insights", {})
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-family:monospace; font-size:0.75rem;">
                        <span style="color:#94A3B8;">Road Surface Occupancy:</span>
                        <span style="color:#38BDF8; font-weight:700;">{insights.get('road_occupancy_pct', 0)}% of frame</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-family:monospace; font-size:0.75rem;">
                        <span style="color:#94A3B8;">Spatial Clustering:</span>
                        <span style="color:#CBD5E1; font-weight:600;">{insights.get('clustering_status', 'N/A')}</span>
                    </div>
                    <div style="background:#060B14; border:1px solid #142033; padding:6px 8px; border-radius:3px; margin-top:8px;">
                        <div style="font-family:monospace; font-size:0.65rem; color:#64748B; margin-bottom:2px;">TACTICAL ADVISORY:</div>
                        <div style="font-family:monospace; font-size:0.72rem; color:#E2E8F0; line-height:1.3;">
                            {insights.get('recommendation', 'Awaiting telemetry.')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='empty-state-box'><div>[NO BEHAVIOR DATA // AWAITING VEHICLE SCAN]</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with low_col3:
            st.markdown(
                """
                <div class="ctrl-panel">
                    <div class="ctrl-header">
                        <span class="ctrl-header-title"><span>📊</span><span>OBJECT TYPE DISTRIBUTION</span></span>
                        <span class="ctrl-tag">DISTRIBUTION</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if curr_res is not None and curr_res["total_count"] > 0:
                counts = curr_res["category_counts"]
                chart_data = {
                    "Category": ["Cars", "Trucks", "Buses", "Bikes"],
                    "Count": [counts["car"], counts["truck"], counts["bus"], counts["motorcycle"]],
                }
                st.bar_chart(chart_data, x="Category", y="Count", height=130)
            else:
                st.markdown("<div class='empty-state-box'><div>[NO OBJECTS DETECTED TO CHART]</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Pollution Monitoring Expandable Tray
        with st.expander("🌱 SENSOR ARRAY: POLLUTION & AIR QUALITY MONITORING [DEMO DATA]", expanded=False):
            st.markdown(
                """
                <div style="background:#060B14; border:1px solid #1E293B; border-left:3px solid #EF4444; padding:8px 12px; margin-bottom:10px; font-family:monospace; font-size:0.72rem;">
                    <span style="color:#EF4444; font-weight:700;">⚠️ NOTICE:</span>
                    <span style="color:#94A3B8;">DEMO SENSOR TELEMETRY — PHYSICAL SENSORS AWAITING FIELD CALIBRATION AND HARDWARE BUS HOOKUP.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            pol_col1, pol_col2, pol_col3, pol_col4 = st.columns(4)
            with pol_col1:
                st.markdown("<div class='sensor-card'><span style='color:#38BDF8; font-weight:700;'>POD-01 (NORTH)</span><br><span style='font-size:1.2rem; font-weight:800; color:#FFFFFF;'>AQI 84</span><br><span style='font-size:0.68rem; color:#94A3B8;'>PM2.5: 28 µg/m³ | PM10: 45 µg/m³</span></div>", unsafe_allow_html=True)
            with pol_col2:
                st.markdown("<div class='sensor-card'><span style='color:#38BDF8; font-weight:700;'>POD-02 (SOUTH)</span><br><span style='font-size:1.2rem; font-weight:800; color:#EF4444;'>AQI 122</span><br><span style='font-size:0.68rem; color:#94A3B8;'>PM2.5: 45 µg/m³ | PM10: 68 µg/m³</span></div>", unsafe_allow_html=True)
            with pol_col3:
                st.markdown("<div class='sensor-card'><span style='color:#38BDF8; font-weight:700;'>POD-03 (EAST)</span><br><span style='font-size:1.2rem; font-weight:800; color:#10B981;'>AQI 65</span><br><span style='font-size:0.68rem; color:#94A3B8;'>PM2.5: 19 µg/m³ | PM10: 32 µg/m³</span></div>", unsafe_allow_html=True)
            with pol_col4:
                st.markdown("<div class='sensor-card'><span style='color:#38BDF8; font-weight:700;'>POD-04 (WEST)</span><br><span style='font-size:1.2rem; font-weight:800; color:#FFFFFF;'>AQI 95</span><br><span style='font-size:0.68rem; color:#94A3B8;'>PM2.5: 34 µg/m³ | PM10: 52 µg/m³</span></div>", unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: CO2 — CIRCULAR LINKED-LIST TRAFFIC RECORD MANAGER
    # ==========================================================================
    with tab_co2:
        st.markdown(
            """
            <div class="ctrl-panel">
                <div class="ctrl-header">
                    <span class="ctrl-header-title">
                        <span>🔄</span>
                        <span>CO2 | CIRCULAR LINKED-LIST TRAFFIC RECORD MANAGER</span>
                    </span>
                    <span class="ctrl-tag" style="border-color:#10B981; color:#10B981;">CYCLIC POINTER STRUCTURE</span>
                </div>
                <div class="dsa-notice">
                    <b>Data Structure:</b> Custom Python <code>CLLNode</code> and <code>CircularLinkedList</code>.
                    Maintains strict pointer invariants where <code>Tail.next = Head</code>. Enables continuous cyclical forward/backward scan auditing.
                </div>
            """,
            unsafe_allow_html=True,
        )

        cll_stats = cll.get_stats()
        cll_total = cll_stats["total_records"]

        # Statistics Row
        st_c1, st_c2, st_c3, st_c4 = st.columns(4)
        with st_c1:
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val'>{cll_total}</div><div class='surv-counter-lbl'>Total Stored Nodes</div></div>", unsafe_allow_html=True)
        with st_c2:
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val' style='color:#38BDF8;'>{cll_stats['head_id']}</div><div class='surv-counter-lbl'>Head Node</div></div>", unsafe_allow_html=True)
        with st_c3:
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val' style='color:#A855F7;'>{cll_stats['tail_id']}</div><div class='surv-counter-lbl'>Tail Node</div></div>", unsafe_allow_html=True)
        with st_c4:
            valid_color = "#10B981" if cll_stats["is_circular_valid"] else "#EF4444"
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val' style='color:{valid_color}; font-size:1.3rem;'>{'VERIFIED' if cll_stats['is_circular_valid'] else 'BROKEN'}</div><div class='surv-counter-lbl'>Tail.next ➔ Head</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        if cll_total == 0:
            st.markdown(
                """
                <div class="empty-state-box">
                    <div style="font-size:1.4rem; margin-bottom:4px;">⭕</div>
                    <div style="color:#38BDF8; font-weight:700;">CIRCULAR LINKED LIST IS EMPTY</div>
                    <div>Upload and scan an image in the "Live Surveillance" tab to automatically append a verified traffic record node.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Clamp navigation index
            idx = st.session_state["cll_nav_index"] % cll_total
            curr_node = cll.get_node_at(idx)
            prev_node = cll.get_prev(curr_node)
            next_node = cll.get_next(curr_node)

            # Circular Pointer Chain Visualization
            prev_id = prev_node.record.scan_id if prev_node else "N/A"
            curr_id = curr_node.record.scan_id if curr_node else "N/A"
            next_id = next_node.record.scan_id if next_node else "N/A"

            st.markdown(
                f"""
                <div class="pointer-chain">
                    <span style="color:#64748B;">CYCLIC TRAVERSAL:</span>
                    <span class="pointer-node">[Prev: {prev_id}]</span>
                    <span class="pointer-arrow">➔</span>
                    <span class="pointer-node pointer-active">[ACTIVE NODE: {curr_id}]</span>
                    <span class="pointer-arrow">➔</span>
                    <span class="pointer-node">[Next: {next_id}]</span>
                    <span style="color:#64748B; font-size:0.68rem; margin-left:auto;">(Index {idx + 1} of {cll_total})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Navigation & Action Controls
            n_c1, n_c2, n_c3, n_c4 = st.columns([22, 28, 22, 28])
            with n_c1:
                if st.button("◀ PREV RECORD", use_container_width=True, key="cll_prev_btn"):
                    st.session_state["cll_nav_index"] = (idx - 1) % cll_total
                    st.rerun()
            with n_c2:
                st.markdown(
                    f"<div style='text-align:center; font-family:monospace; font-size:0.85rem; font-weight:700; color:#38BDF8; padding:6px;'>"
                    f"Node: {curr_id} ({idx + 1}/{cll_total})"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with n_c3:
                if st.button("NEXT RECORD ▶", use_container_width=True, key="cll_next_btn"):
                    st.session_state["cll_nav_index"] = (idx + 1) % cll_total
                    st.rerun()
            with n_c4:
                if st.button(f"🗑️ DELETE {curr_id}", use_container_width=True, key="cll_del_btn"):
                    cll.delete_by_id(curr_id)
                    st.session_state["cll_nav_index"] = max(0, idx - 1)
                    st.success(f"Safely unlinked and deleted {curr_id}. Circular pointers preserved.")
                    time.sleep(0.5)
                    st.rerun()

            # Active Record Display Card
            rec = curr_node.record
            st.markdown(
                f"""
                <div style="background:#060B14; border:1px solid #1E2D4A; border-radius:4px; padding:12px 16px; margin: 12px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #142033; padding-bottom:6px; margin-bottom:8px;">
                        <span style="font-family:monospace; font-size:1.1rem; font-weight:800; color:#38BDF8;">{rec.scan_id}</span>
                        <span class="ctrl-tag" style="border-color:#10B981; color:#10B981;">{rec.congestion_level.upper()} DENSITY</span>
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; font-family:monospace; font-size:0.75rem;">
                        <div><span style="color:#64748B;">TIMESTAMP:</span><br><span style="color:#FFFFFF;">{rec.timestamp}</span></div>
                        <div><span style="color:#64748B;">SOURCE IMAGE:</span><br><span style="color:#CBD5E1;">{rec.image_filename}</span></div>
                        <div><span style="color:#64748B;">TOTAL VEHICLES:</span><br><span style="color:#00E5FF; font-weight:700; font-size:0.95rem;">{rec.total_vehicles}</span></div>
                        <div><span style="color:#64748B;">CONGESTION SCORE:</span><br><span style="color:#FFAA00; font-weight:700; font-size:0.95rem;">{rec.congestion_score}</span></div>
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:6px; margin-top:10px; text-align:center; font-family:monospace; font-size:0.72rem;">
                        <div style="background:#091220; padding:4px; border-radius:2px;"><span style="color:#00E5FF;">Cars:</span> {rec.car_count}</div>
                        <div style="background:#091220; padding:4px; border-radius:2px;"><span style="color:#22C55E;">Trucks:</span> {rec.truck_count}</div>
                        <div style="background:#091220; padding:4px; border-radius:2px;"><span style="color:#D946EF;">Buses:</span> {rec.bus_count}</div>
                        <div style="background:#091220; padding:4px; border-radius:2px;"><span style="color:#FFAA00;">Bikes:</span> {rec.motorcycle_count}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tabular inspection of all circular records
            with st.expander("📋 FULL CIRCULAR LINKED-LIST REGISTRY (TRAVERSAL ORDER)", expanded=False):
                records_list = [r.to_dict() for r in cll.to_list()]
                st.dataframe(records_list, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # TAB 3: CO3 — AVL BALANCED TREE FOR CONGESTION TRENDS & SEARCHING
    # ==========================================================================
    with tab_co3:
        st.markdown(
            """
            <div class="ctrl-panel">
                <div class="ctrl-header">
                    <span class="ctrl-header-title">
                        <span>🌳</span>
                        <span>CO3 | AVL BALANCED TREE CONGESTION ANALYTICS</span>
                    </span>
                    <span class="ctrl-tag" style="border-color:#38BDF8;">SELF-BALANCING BST</span>
                </div>
                <div class="dsa-notice">
                    <b>Data Structure:</b> Custom Python <code>AVLNode</code> and <code>AVLTree</code>.
                    Keys ordered deterministically by <code>(congestion_score, scan_id)</code>.
                    Enforces strict balance factor <code>|height(left) - height(right)| ≤ 1</code> via LL, RR, LR, and RL rotations.
                </div>
            """,
            unsafe_allow_html=True,
        )

        avl_stats = avl.get_tree_stats()
        avl_total = avl_stats["total_nodes"]

        # Tree Diagnostics Row
        a_c1, a_c2, a_c3, a_c4, a_c5 = st.columns(5)
        with a_c1:
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val'>{avl_total}</div><div class='surv-counter-lbl'>Total Nodes</div></div>", unsafe_allow_html=True)
        with a_c2:
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val' style='color:#38BDF8;'>{avl_stats['tree_height']}</div><div class='surv-counter-lbl'>Tree Height</div></div>", unsafe_allow_html=True)
        with a_c3:
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val' style='color:#A855F7;'>{avl_stats['root_balance_factor']}</div><div class='surv-counter-lbl'>Root Balance Factor</div></div>", unsafe_allow_html=True)
        with a_c4:
            b_col = "#10B981" if avl_stats["is_strictly_balanced"] else "#EF4444"
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val' style='color:{b_col}; font-size:1.3rem;'>{'STRICT' if avl_stats['is_strictly_balanced'] else 'UNBALANCED'}</div><div class='surv-counter-lbl'>AVL Invariant</div></div>", unsafe_allow_html=True)
        with a_c5:
            rots = avl_stats["rotations"]
            rot_summary = f"LL:{rots['LL']} RR:{rots['RR']} LR:{rots['LR']} RL:{rots['RL']}"
            st.markdown(f"<div class='surv-counter-box'><div class='surv-counter-val' style='font-size:0.95rem; color:#FFAA00; line-height:1.7;'>{rot_summary}</div><div class='surv-counter-lbl'>Rotations Applied</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        if avl_total == 0:
            st.markdown(
                """
                <div class="empty-state-box">
                    <div style="font-size:1.4rem; margin-bottom:4px;">🌲</div>
                    <div style="color:#38BDF8; font-weight:700;">AVL BALANCED TREE IS EMPTY</div>
                    <div>Perform vehicle scans in the "Live Surveillance" tab to populate and index congestion records in logarithmic order.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Search & Retrieval Interface
            search_col1, search_col2 = st.columns(2)

            with search_col1:
                st.markdown("<div style='font-family:monospace; font-size:0.75rem; color:#38BDF8; font-weight:700; margin-bottom:4px;'>🔍 EXACT SEARCH BY SCAN ID:</div>", unsafe_allow_html=True)
                all_ids = [r.scan_id for r in avl.in_order_traversal()]
                selected_search_id = st.selectbox("Select Scan ID to Search", all_ids, key="avl_search_select")
                matched_record = avl.search_by_id(selected_search_id)

                if matched_record:
                    st.markdown(
                        f"""
                        <div style="background:#060B14; border:1px solid #142033; padding:8px 12px; border-radius:3px; font-family:monospace; font-size:0.75rem;">
                            <span style="color:#10B981; font-weight:700;">MATCH FOUND (O(log n)):</span> {matched_record.scan_id}<br>
                            <span style="color:#94A3B8;">Congestion Score:</span> {matched_record.congestion_score} |
                            <span style="color:#94A3B8;">Level:</span> {matched_record.congestion_level}<br>
                            <span style="color:#94A3B8;">Timestamp:</span> {matched_record.timestamp}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with search_col2:
                st.markdown("<div style='font-family:monospace; font-size:0.75rem; color:#38BDF8; font-weight:700; margin-bottom:4px;'>🎯 RANGE SEARCH BY CONGESTION SCORE:</div>", unsafe_allow_html=True)
                min_s = float(avl_stats["min_score"])
                max_s = float(avl_stats["max_score"])
                if min_s == max_s:
                    max_s = min_s + 5.0
                score_range = st.slider("Congestion Score Range", min_value=0.0, max_value=max(max_s, 25.0), value=(0.0, max(max_s, 25.0)), step=1.0)
                range_matches = avl.search_by_score_range(score_range[0], score_range[1])
                st.caption(f"Discovered {len(range_matches)} records within score range [{score_range[0]} - {score_range[1]}].")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Historical Congestion Trend Analysis (From Scanned Records)
            st.markdown("<div class='ctrl-header'><span class='ctrl-header-title'><span>📈</span><span>CONGESTION TREND ANALYSIS & CHARTS</span></span></div>", unsafe_allow_html=True)
            trend_info = avl.compute_trend_analysis()

            t_c1, t_c2 = st.columns([40, 60])
            with t_c1:
                t_color = trend_info.get("color", "#64748B")
                st.markdown(
                    f"""
                    <div style="background:#060B14; border:1px solid #1E2D4A; border-left:3px solid {t_color}; padding:10px 14px; border-radius:3px; height:100%;">
                        <div style="font-family:monospace; font-size:0.7rem; color:#64748B;">HISTORICAL TELEMETRY DIRECTION:</div>
                        <div style="font-family:monospace; font-size:1.15rem; font-weight:800; color:{t_color}; margin: 4px 0;">{trend_info['direction']}</div>
                        <div style="font-family:monospace; font-size:0.72rem; color:#94A3B8;">Slope Delta: {trend_info.get('delta', 0.0):+0.2f} | Mean Score: {trend_info.get('avg_score', 0.0)}</div>
                        <div style="font-family:monospace; font-size:0.65rem; color:#64748B; margin-top:8px; line-height:1.3;">{trend_info['note']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with t_c2:
                # Congestion score progression line chart across scan timestamps
                chrono_records = sorted(avl.in_order_traversal(), key=lambda r: r.scan_id)
                chart_dict = {
                    "Scan ID": [r.scan_id for r in chrono_records],
                    "Congestion Score": [r.congestion_score for r in chrono_records],
                }
                st.line_chart(chart_dict, x="Scan ID", y="Congestion Score", height=150)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # In-Order Sorted Records (Ascending by Congestion Score)
            with st.expander("📊 AVL IN-ORDER TRAVERSAL (SORTED BY CONGESTION SCORE)", expanded=True):
                sorted_records = avl.in_order_traversal()
                sorted_data = [
                    {
                        "Sort Rank": i + 1,
                        "Congestion Score": r.congestion_score,
                        "Scan ID": r.scan_id,
                        "Level": r.congestion_level,
                        "Total Vehicles": r.total_vehicles,
                        "Cars": r.car_count,
                        "Trucks": r.truck_count,
                        "Buses": r.bus_count,
                        "Bikes": r.motorcycle_count,
                        "Timestamp": r.timestamp,
                    }
                    for i, r in enumerate(sorted_records)
                ]
                st.dataframe(sorted_data, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # TAB 4: CO5 — ROAD NETWORK, ADJACENCY LIST, & DFS ALTERNATIVE ROUTES
    # ==========================================================================
    with tab_co5:
        st.markdown(
            """
            <div class="ctrl-panel">
                <div class="ctrl-header">
                    <span class="ctrl-header-title">
                        <span>🗺️</span>
                        <span>CO5 | ROAD NETWORK GRAPH & DFS ALTERNATIVE ROUTE ENGINE</span>
                    </span>
                    <span class="ctrl-tag" style="border-color:#F59E0B; color:#F59E0B;">ADJACENCY LIST & DFS</span>
                </div>
                <div class="dsa-notice" style="border-left-color:#F59E0B;">
                    <b>DEMONSTRATION ROAD NETWORK — SAMPLE TOPOLOGY.</b><br>
                    <b>Notice:</b> Depth-First Search (DFS) explores possible candidate routes; DFS alone does not guarantee the globally shortest or optimal route.
                    The selected route is the lowest-cost route among candidate paths explored within the configured search depth.
                    Real-world traffic conditions are not used unless an external live sensor API is connected.
                </div>
            """,
            unsafe_allow_html=True,
        )

        all_intersections = graph.get_intersections()

        # Origin / Destination & Weights Configuration
        ctl_c1, ctl_c2, ctl_c3, ctl_c4 = st.columns([26, 26, 24, 24])

        with ctl_c1:
            start_j = st.selectbox("Origin Intersection", all_intersections, index=0, key="graph_start_sel")
        with ctl_c2:
            dest_j = st.selectbox("Destination Intersection", all_intersections, index=min(6, len(all_intersections) - 1), key="graph_dest_sel")
        with ctl_c3:
            w_dist = st.slider("Distance Weight (w_dist)", 0.0, 1.0, 0.5, 0.05, help="Relative weight for physical road distance.")
        with ctl_c4:
            w_cong = st.slider("Congestion Weight (w_cong)", 0.0, 1.0, 0.5, 0.05, help="Relative penalty for congested road corridors.")

        # Trigger DFS Route Search
        search_routes_btn = st.button("🔍 EXPLORE ALTERNATIVE ROUTES (DFS)", use_container_width=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if start_j == dest_j:
            st.warning("Origin and Destination are identical. Please select distinct intersections.")
        else:
            # Execute DFS route search
            candidate_routes = graph.find_routes_dfs(start_j, dest_j, max_routes=10, max_depth=7)

            if not candidate_routes:
                st.markdown(
                    """
                    <div class="empty-state-box">
                        <div>[NO FEASIBLE ROUTES DISCOVERED VIA DFS SEARCH LIMITS]</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                best_route, evaluated_routes = graph.evaluate_routes(
                    candidate_routes,
                    weight_distance=w_dist,
                    weight_congestion=w_cong,
                )

                # Best Route Summary Box
                if best_route:
                    st.markdown(
                        f"""
                        <div class="route-card route-card-best">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span style="font-family:monospace; font-size:0.85rem; font-weight:800; color:#00E5FF;">
                                    🏆 SELECTED LOWEST-COST ROUTE (AMONG DFS CANDIDATES)
                                </span>
                                <span class="ctrl-tag" style="border-color:#00E5FF; color:#00E5FF;">
                                    COMPOSITE COST: {best_route['composite_cost']}
                                </span>
                            </div>
                            <div style="font-family:monospace; font-size:1.0rem; font-weight:700; color:#FFFFFF; margin-bottom:6px;">
                                {' ➔ '.join(best_route['path'])}
                            </div>
                            <div style="display:flex; gap:20px; font-family:monospace; font-size:0.75rem; color:#94A3B8;">
                                <span>Total Distance: <b style="color:#FFFFFF;">{best_route['total_distance_km']} km</b></span>
                                <span>Congestion Cost: <b style="color:#FFAA00;">{best_route['total_congestion_cost']}</b></span>
                                <span>Intersections: <b style="color:#38BDF8;">{len(best_route['path'])}</b></span>
                                <span>Hops: <b style="color:#CBD5E1;">{best_route['hops']}</b></span>
                            </div>
                            <div style="font-family:monospace; font-size:0.65rem; color:#64748B; margin-top:6px;">
                                Formula: Cost = ({w_dist:.2f} × {best_route['total_distance_km']} km) + ({w_cong:.2f} × {best_route['total_congestion_cost']})
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # All Candidate Routes Table
                st.markdown("<div class='ctrl-header'><span class='ctrl-header-title'><span>📋</span><span>DFS CANDIDATE ROUTES COMPARISON</span></span></div>", unsafe_allow_html=True)
                table_rows = []
                for i, r in enumerate(evaluated_routes, start=1):
                    table_rows.append({
                        "Rank": f"#{i} {'(Selected)' if i==1 else ''}",
                        "Composite Cost": r["composite_cost"],
                        "Distance (km)": r["total_distance_km"],
                        "Congestion Cost": r["total_congestion_cost"],
                        "Hops": r["hops"],
                        "Full Route Path": " ➔ ".join(r["path"]),
                    })
                st.dataframe(table_rows, use_container_width=True)

        # Adjacency List Network Representation
        with st.expander("🌐 VIEW ADJACENCY LIST & ROAD NETWORK GRAPH TOPOLOGY", expanded=False):
            adj_rep = graph.get_adjacency_display()
            for u, edges in adj_rep.items():
                meta = graph.intersections.get(u, {})
                label = meta.get("label", u)
                edge_desc = ", ".join([f"{e['to']} via {e['road']} ({e['distance_km']}km, cost:{e['congestion_cost']})" for e in edges])
                st.markdown(
                    f"<div style='font-family:monospace; font-size:0.72rem; margin-bottom:4px;'>"
                    f"<span style='color:#38BDF8; font-weight:700;'>[{u}] {label}:</span> "
                    f"<span style='color:#CBD5E1;'>{edge_desc}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)


def create_synthetic_road_sample() -> Image.Image:
    """Creates a sample highway scene with asphalt road, lane dividers, and horizon."""
    w, h = 960, 540
    img = np.full((h, w, 3), (25, 30, 38), dtype=np.uint8)

    # Road edges
    cv2.line(img, (0, 110), (w, 110), (70, 75, 85), 3)
    cv2.line(img, (0, h - 70), (w, h - 70), (70, 75, 85), 3)

    # Dashed lane markings
    for x in range(0, w, 55):
        cv2.line(img, (x, int(h * 0.42)), (x + 32, int(h * 0.42)), (200, 200, 200), 2)
        cv2.line(img, (x, int(h * 0.68)), (x + 32, int(h * 0.68)), (200, 200, 200), 2)

    # Horizon sky
    img[:110, :] = (14, 18, 26)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
