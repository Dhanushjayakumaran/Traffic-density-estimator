"""
dsa_structures.py - Core Data Structures and Algorithms Module
Implements:
1. CO2: Circular Linked-List Traffic Record Manager (custom Node & CircularLinkedList)
2. CO3: AVL Balanced Binary Search Tree (custom AVLNode, AVLTree, all 4 rotations LL/RR/LR/RL)
3. CO5: Road Network Graph & DFS Route Cost Evaluator (custom Adjacency List & DFS with backtracking)
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import datetime
import math


# ==============================================================================
# COMMON TRAFFIC RECORD MODEL
# ==============================================================================

@dataclass
class TrafficRecord:
    """Unified traffic scan record shared between circular linked list and AVL tree."""
    scan_id: str
    timestamp: str
    car_count: int
    truck_count: int
    bus_count: int
    motorcycle_count: int
    total_vehicles: int
    congestion_level: str  # "Low", "Medium", "High"
    congestion_score: float  # Numerical metric for ordering/searching
    image_filename: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Scan ID": self.scan_id,
            "Timestamp": self.timestamp,
            "Total Vehicles": self.total_vehicles,
            "Congestion Score": self.congestion_score,
            "Congestion Level": self.congestion_level,
            "Cars": self.car_count,
            "Trucks": self.truck_count,
            "Buses": self.bus_count,
            "Motorcycles": self.motorcycle_count,
            "Source Image": self.image_filename,
        }


# ==============================================================================
# CO2: CIRCULAR SINGLY LINKED LIST
# ==============================================================================

class CLLNode:
    """Node for circular singly linked list."""
    def __init__(self, record: TrafficRecord):
        self.record: TrafficRecord = record
        self.next: Optional["CLLNode"] = None


class CircularLinkedList:
    """
    Genuine Circular Singly Linked List for sequential traffic scan navigation.
    The last node's 'next' pointer strictly references the 'head' node.
    """
    def __init__(self):
        self.head: Optional[CLLNode] = None
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self.head is None

    def append(self, record: TrafficRecord) -> None:
        """Insert record at the end of circular list, maintaining tail.next -> head."""
        new_node = CLLNode(record)
        if self.head is None:
            self.head = new_node
            new_node.next = self.head
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
            new_node.next = self.head
        self._size += 1

    def prepend(self, record: TrafficRecord) -> None:
        """Insert record at the beginning of circular list, updating tail.next to new head."""
        new_node = CLLNode(record)
        if self.head is None:
            self.head = new_node
            new_node.next = self.head
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
            new_node.next = self.head
            self.head = new_node
        self._size += 1

    def delete_by_id(self, scan_id: str) -> bool:
        """
        Safely delete a record by scan_id from the circular linked list.
        Returns True if deleted, False if not found or list is empty.
        """
        if self.head is None:
            return False

        current = self.head
        prev = None

        # Case 1: Head node matches scan_id
        if current.record.scan_id == scan_id:
            # Subcase 1a: Only 1 node in the circular list
            if current.next == self.head:
                self.head = None
                self._size = 0
                return True
            # Subcase 1b: More than 1 node, find tail to update tail.next
            tail = self.head
            while tail.next != self.head:
                tail = tail.next
            self.head = current.next
            tail.next = self.head
            self._size -= 1
            return True

        # Case 2: Node is elsewhere in the circular list
        prev = current
        current = current.next
        while current != self.head:
            if current.record.scan_id == scan_id:
                prev.next = current.next
                self._size -= 1
                return True
            prev = current
            current = current.next

        return False

    def get_node_at(self, index: int) -> Optional[CLLNode]:
        """Retrieve node at logical index (0 to size-1)."""
        if self.head is None or index < 0 or index >= self._size:
            return None
        current = self.head
        for _ in range(index):
            current = current.next
        return current

    def get_next(self, node: CLLNode) -> Optional[CLLNode]:
        """Return next node in circular loop."""
        if self.head is None or node is None:
            return None
        return node.next

    def get_prev(self, node: CLLNode) -> Optional[CLLNode]:
        """
        In a circular singly linked list, find the previous node by traversing
        until current.next == node.
        """
        if self.head is None or node is None:
            return None
        current = self.head
        while current.next != node:
            current = current.next
            if current == self.head:  # Loop completed without finding
                return None
        return current

    def search_by_id(self, scan_id: str) -> Optional[CLLNode]:
        """Search node by scan_id using circular traversal."""
        if self.head is None:
            return None
        current = self.head
        while True:
            if current.record.scan_id == scan_id:
                return current
            current = current.next
            if current == self.head:
                break
        return None

    def to_list(self) -> List[TrafficRecord]:
        """Traverse the circle once starting from head, returning records as a list."""
        records: List[TrafficRecord] = []
        if self.head is None:
            return records
        current = self.head
        while True:
            records.append(current.record)
            current = current.next
            if current == self.head:
                break
        return records

    def get_stats(self) -> Dict[str, Any]:
        """Verification statistics for control room dashboard."""
        if self.head is None:
            return {
                "total_records": 0,
                "is_empty": True,
                "head_id": "None",
                "tail_id": "None",
                "is_circular_valid": True,
            }
        tail = self.head
        while tail.next != self.head:
            tail = tail.next

        return {
            "total_records": self._size,
            "is_empty": False,
            "head_id": self.head.record.scan_id,
            "tail_id": tail.record.scan_id,
            "is_circular_valid": (tail.next == self.head),
        }


# ==============================================================================
# CO3: AVL BALANCED BINARY SEARCH TREE
# ==============================================================================

class AVLNode:
    """Node for AVL Balanced Search Tree."""
    def __init__(self, record: TrafficRecord):
        self.record: TrafficRecord = record
        # Primary key: (congestion_score, scan_id) for strict deterministic ordering
        self.key: Tuple[float, str] = (record.congestion_score, record.scan_id)
        self.height: int = 1
        self.left: Optional["AVLNode"] = None
        self.right: Optional["AVLNode"] = None


class AVLTree:
    """
    Genuine AVL Self-Balancing Binary Search Tree.
    Maintains O(log n) search, insertion, and ordered traversal.
    Supports all four rotation cases: LL, RR, LR, RL.
    """
    def __init__(self):
        self.root: Optional[AVLNode] = None
        self._count: int = 0
        self.rotations_counter: Dict[str, int] = {"LL": 0, "RR": 0, "LR": 0, "RL": 0}

    def __len__(self) -> int:
        return self._count

    def get_height(self, node: Optional[AVLNode]) -> int:
        return node.height if node else 0

    def get_balance(self, node: Optional[AVLNode]) -> int:
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def _update_height(self, node: AVLNode) -> None:
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

    def _rotate_right(self, y: AVLNode) -> AVLNode:
        """
        Right rotation on node y (LL case).
             y                     x
            / \                   / \
           x   T3   ===>        T1   y
          / \                       / \
         T1  T2                    T2  T3
        """
        x = y.left
        T2 = x.right

        # Perform rotation
        x.right = y
        y.left = T2

        # Update heights
        self._update_height(y)
        self._update_height(x)

        return x

    def _rotate_left(self, x: AVLNode) -> AVLNode:
        """
        Left rotation on node x (RR case).
           x                         y
          / \                       / \
         T1  y        ===>         x   T3
            / \                   / \
           T2  T3                T1  T2
        """
        y = x.right
        T2 = y.left

        # Perform rotation
        y.left = x
        x.right = T2

        # Update heights
        self._update_height(x)
        self._update_height(y)

        return y

    def insert(self, record: TrafficRecord) -> None:
        """Public insertion API."""
        key = (record.congestion_score, record.scan_id)
        self.root = self._insert_node(self.root, record, key)
        self._count += 1

    def _insert_node(self, node: Optional[AVLNode], record: TrafficRecord, key: Tuple[float, str]) -> AVLNode:
        # Step 1: Standard BST recursive insertion
        if not node:
            return AVLNode(record)

        if key < node.key:
            node.left = self._insert_node(node.left, record, key)
        elif key > node.key:
            node.right = self._insert_node(node.right, record, key)
        else:
            # Duplicate key: update record
            node.record = record
            return node

        # Step 2: Update height of ancestor node
        self._update_height(node)

        # Step 3: Compute balance factor to check if node became unbalanced
        balance = self.get_balance(node)

        # Case 1: Left-Left (LL) -> Right rotation on node
        if balance > 1 and key < node.left.key:
            self.rotations_counter["LL"] += 1
            return self._rotate_right(node)

        # Case 2: Right-Right (RR) -> Left rotation on node
        if balance < -1 and key > node.right.key:
            self.rotations_counter["RR"] += 1
            return self._rotate_left(node)

        # Case 3: Left-Right (LR) -> Left rotation on left child, then Right rotation on node
        if balance > 1 and key > node.left.key:
            self.rotations_counter["LR"] += 1
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Case 4: Right-Left (RL) -> Right rotation on right child, then Left rotation on node
        if balance < -1 and key < node.right.key:
            self.rotations_counter["RL"] += 1
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def in_order_traversal(self) -> List[TrafficRecord]:
        """In-order traversal yielding records sorted by congestion score (ascending)."""
        result: List[TrafficRecord] = []
        self._in_order(self.root, result)
        return result

    def _in_order(self, node: Optional[AVLNode], acc: List[TrafficRecord]) -> None:
        if node:
            self._in_order(node.left, acc)
            acc.append(node.record)
            self._in_order(node.right, acc)

    def search_by_id(self, scan_id: str) -> Optional[TrafficRecord]:
        """Search tree for record with given scan_id."""
        return self._search_id_helper(self.root, scan_id)

    def _search_id_helper(self, node: Optional[AVLNode], scan_id: str) -> Optional[TrafficRecord]:
        if not node:
            return None
        if node.record.scan_id == scan_id:
            return node.record
        found = self._search_id_helper(node.left, scan_id)
        if found:
            return found
        return self._search_id_helper(node.right, scan_id)

    def search_by_score_range(self, min_score: float, max_score: float) -> List[TrafficRecord]:
        """Find all records where min_score <= congestion_score <= max_score."""
        matches: List[TrafficRecord] = []
        self._range_search_helper(self.root, min_score, max_score, matches)
        return matches

    def _range_search_helper(self, node: Optional[AVLNode], min_s: float, max_s: float, acc: List[TrafficRecord]) -> None:
        if not node:
            return
        score = node.record.congestion_score
        if score > min_s:
            self._range_search_helper(node.left, min_s, max_s, acc)
        if min_s <= score <= max_s:
            acc.append(node.record)
        if score < max_s:
            self._range_search_helper(node.right, min_s, max_s, acc)

    def get_tree_stats(self) -> Dict[str, Any]:
        """Diagnostic stats for tree balance verification."""
        height = self.get_height(self.root)
        balance = self.get_balance(self.root)
        is_balanced = self._check_is_balanced(self.root)
        records = self.in_order_traversal()
        min_score = records[0].congestion_score if records else 0.0
        max_score = records[-1].congestion_score if records else 0.0

        return {
            "total_nodes": self._count,
            "tree_height": height,
            "root_balance_factor": balance,
            "is_strictly_balanced": is_balanced,
            "min_score": min_score,
            "max_score": max_score,
            "rotations": self.rotations_counter.copy(),
        }

    def _check_is_balanced(self, node: Optional[AVLNode]) -> bool:
        if not node:
            return True
        b = self.get_balance(node)
        if abs(b) > 1:
            return False
        return self._check_is_balanced(node.left) and self._check_is_balanced(node.right)

    def compute_trend_analysis(self) -> Dict[str, Any]:
        """
        Analyze chronological trend of congestion scores from records.
        Explicitly notes that this reflects past scanned data, not forward predictions.
        """
        records = self.in_order_traversal()
        # Sort chronologically by scan_id / timestamp
        chrono_records = sorted(records, key=lambda r: r.scan_id)
        count = len(chrono_records)

        if count < 2:
            return {
                "status": "Insufficient Data",
                "direction": "Unknown",
                "delta": 0.0,
                "avg_score": chrono_records[0].congestion_score if count == 1 else 0.0,
                "note": "AVL tree organizes historical records; at least 2 scans are needed for trend slope.",
            }

        first_half = chrono_records[:count // 2]
        second_half = chrono_records[count // 2:]

        avg_first = sum(r.congestion_score for r in first_half) / len(first_half)
        avg_second = sum(r.congestion_score for r in second_half) / len(second_half)
        delta = round(avg_second - avg_first, 2)

        if delta > 1.5:
            direction = "Increasing Congestion"
            color = "#EF4444"
        elif delta < -1.5:
            direction = "Decreasing Congestion"
            color = "#10B981"
        else:
            direction = "Stable Traffic Flow"
            color = "#38BDF8"

        return {
            "status": "Active Analysis",
            "direction": direction,
            "delta": delta,
            "avg_score": round(sum(r.congestion_score for r in chrono_records) / count, 1),
            "color": color,
            "note": "Notice: The AVL tree organizes and searches historical scan data; it does not predict future traffic unless a forecasting model is separately implemented.",
        }


# ==============================================================================
# CO5: ROAD NETWORK GRAPH, ADJACENCY LIST, & DFS ROUTE EVALUATOR
# ==============================================================================

@dataclass
class RoadEdge:
    """Directed edge representing a road segment between two intersections."""
    destination: str
    distance_km: float
    congestion_cost: float  # Multiplier or latency penalty (1.0 = clear, 5.0+ = severe delay)
    road_name: str


class RoadGraph:
    """
    Road Network represented via an Adjacency List.
    Uses Depth-First Search (DFS) with cycle avoidance and backtracking
    to discover candidate paths and evaluate composite route costs.
    """
    def __init__(self):
        # Adjacency list: vertex -> list of RoadEdge
        self.adj_list: Dict[str, List[RoadEdge]] = {}
        # Metadata for intersections
        self.intersections: Dict[str, Dict[str, Any]] = {}

    def add_intersection(self, node_id: str, label: str, sector: str, coords: Tuple[int, int]) -> None:
        if node_id not in self.adj_list:
            self.adj_list[node_id] = []
        self.intersections[node_id] = {
            "label": label,
            "sector": sector,
            "coords": coords,
        }

    def add_road(self, u: str, v: str, distance_km: float, congestion_cost: float, road_name: str, bidirectional: bool = True) -> None:
        if u not in self.adj_list:
            self.adj_list[u] = []
        if v not in self.adj_list:
            self.adj_list[v] = []

        self.adj_list[u].append(RoadEdge(destination=v, distance_km=distance_km, congestion_cost=congestion_cost, road_name=road_name))
        if bidirectional:
            self.adj_list[v].append(RoadEdge(destination=u, distance_km=distance_km, congestion_cost=congestion_cost, road_name=road_name))

    def get_intersections(self) -> List[str]:
        return sorted(list(self.adj_list.keys()))

    def get_adjacency_display(self) -> Dict[str, List[Dict[str, Any]]]:
        """Readable dictionary representation of adjacency list."""
        rep: Dict[str, List[Dict[str, Any]]] = {}
        for u, edges in self.adj_list.items():
            rep[u] = [
                {
                    "to": e.destination,
                    "distance_km": e.distance_km,
                    "congestion_cost": e.congestion_cost,
                    "road": e.road_name,
                }
                for e in edges
            ]
        return rep

    def find_routes_dfs(
        self,
        start: str,
        destination: str,
        max_routes: int = 15,
        max_depth: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Depth-First Search (DFS) with backtracking to discover candidate routes.
        Avoids cycles by maintaining a visited set for the current branch.
        """
        if start not in self.adj_list or destination not in self.adj_list:
            return []

        all_routes: List[Dict[str, Any]] = []
        visited = set([start])

        def dfs_recursive(current: str, path: List[str], edges_used: List[RoadEdge], depth: int):
            if len(all_routes) >= max_routes or depth > max_depth:
                return

            if current == destination:
                total_dist = sum(e.distance_km for e in edges_used)
                total_cong = sum(e.congestion_cost for e in edges_used)
                all_routes.append({
                    "path": list(path),
                    "edges": [
                        {
                            "from": path[i],
                            "to": path[i+1],
                            "road": e.road_name,
                            "dist_km": e.distance_km,
                            "cong_cost": e.congestion_cost,
                        }
                        for i, e in enumerate(edges_used)
                    ],
                    "total_distance_km": round(total_dist, 2),
                    "total_congestion_cost": round(total_cong, 2),
                    "hops": len(edges_used),
                })
                return

            for edge in self.adj_list.get(current, []):
                neighbor = edge.destination
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    edges_used.append(edge)

                    dfs_recursive(neighbor, path, edges_used, depth + 1)

                    # Backtracking
                    edges_used.pop()
                    path.pop()
                    visited.remove(neighbor)

        dfs_recursive(start, [start], [], 0)
        return all_routes

    def evaluate_routes(
        self,
        routes: List[Dict[str, Any]],
        weight_distance: float = 0.5,
        weight_congestion: float = 0.5,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Calculate composite route cost using user-configured distance vs congestion weighting:
        Composite Cost = (w_dist * Total Distance) + (w_cong * Total Congestion Cost)
        Ranks all candidate routes and returns (best_route, ranked_routes).
        """
        if not routes:
            return None, []

        evaluated = []
        for r in routes:
            comp_cost = round((weight_distance * r["total_distance_km"]) + (weight_congestion * r["total_congestion_cost"]), 3)
            r_copy = dict(r)
            r_copy["composite_cost"] = comp_cost
            evaluated.append(r_copy)

        # Sort candidate routes by lowest composite cost
        evaluated.sort(key=lambda item: item["composite_cost"])
        best_route = evaluated[0] if evaluated else None
        return best_route, evaluated


def create_sample_road_network() -> RoadGraph:
    """
    Builds a realistic 7-junction demonstration road network with multiple alternative routes.
    Clearly designated as DEMONSTRATION DATA.
    """
    g = RoadGraph()

    # Intersections (Vertices)
    g.add_intersection("J1_GATEWAY", "North Highway Gateway", "Sector 01", (100, 50))
    g.add_intersection("J2_TECHPARK", "Tech Park Interchange", "Sector 02", (250, 100))
    g.add_intersection("J3_CITYCENTER", "City Center Square", "Sector 03", (200, 250))
    g.add_intersection("J4_INDUSTRIAL", "Industrial Boulevard", "Sector 04", (80, 220))
    g.add_intersection("J5_RIVERFRONT", "Riverfront Expressway", "Sector 05", (350, 200))
    g.add_intersection("J6_FLYOVER", "Airport Flyover Junction", "Sector 06", (320, 320))
    g.add_intersection("J7_TERMINAL", "South Logistics Terminal", "Sector 07", (200, 380))

    # Connecting Roads (Edges: distance in km, congestion cost multiplier)
    # J1 Connections
    g.add_road("J1_GATEWAY", "J2_TECHPARK", distance_km=4.2, congestion_cost=2.8, road_name="North Radial Ave")
    g.add_road("J1_GATEWAY", "J4_INDUSTRIAL", distance_km=5.6, congestion_cost=1.4, road_name="Outer Ring West")

    # J2 Connections
    g.add_road("J2_TECHPARK", "J3_CITYCENTER", distance_km=3.1, congestion_cost=4.5, road_name="Central Metro Way")
    g.add_road("J2_TECHPARK", "J5_RIVERFRONT", distance_km=6.0, congestion_cost=1.2, road_name="River Bypass North")

    # J4 Connections
    g.add_road("J4_INDUSTRIAL", "J3_CITYCENTER", distance_km=3.8, congestion_cost=3.2, road_name="Commercial Link")
    g.add_road("J4_INDUSTRIAL", "J7_TERMINAL", distance_km=7.4, congestion_cost=1.8, road_name="Heavy Freight Corridor")

    # J3 Connections
    g.add_road("J3_CITYCENTER", "J6_FLYOVER", distance_km=4.0, congestion_cost=3.9, road_name="Boulevard Spine")
    g.add_road("J3_CITYCENTER", "J7_TERMINAL", distance_km=5.0, congestion_cost=4.2, road_name="South Arterial")

    # J5 Connections
    g.add_road("J5_RIVERFRONT", "J6_FLYOVER", distance_km=3.5, congestion_cost=1.5, road_name="Express Bypass South")

    # J6 Connections
    g.add_road("J6_FLYOVER", "J7_TERMINAL", distance_km=2.8, congestion_cost=2.1, road_name="Terminal Connector")

    return g
