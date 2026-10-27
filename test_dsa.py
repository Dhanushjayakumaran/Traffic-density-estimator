"""
test_dsa.py - Verification and Unit Tests for CO2, CO3, and CO5 DSA components
"""

import unittest
from dsa_structures import (
    TrafficRecord,
    CircularLinkedList,
    AVLTree,
    RoadGraph,
    create_sample_road_network,
)


class TestCircularLinkedList(unittest.TestCase):
    """Test suite for CO2: Circular Linked-List Traffic Record Manager."""

    def setUp(self):
        self.cll = CircularLinkedList()
        self.r1 = TrafficRecord("SCAN-001", "12:00:00", 4, 1, 0, 2, 7, "Medium", 7.0, "road1.png")
        self.r2 = TrafficRecord("SCAN-002", "12:05:00", 2, 0, 0, 1, 3, "Low", 3.0, "road2.png")
        self.r3 = TrafficRecord("SCAN-003", "12:10:00", 10, 3, 2, 4, 19, "High", 19.0, "road3.png")

    def test_empty_list_handling(self):
        self.assertTrue(self.cll.is_empty())
        self.assertEqual(len(self.cll), 0)
        self.assertIsNone(self.cll.get_node_at(0))
        self.assertFalse(self.cll.delete_by_id("SCAN-999"))
        self.assertEqual(len(self.cll.to_list()), 0)
        stats = self.cll.get_stats()
        self.assertTrue(stats["is_empty"])
        self.assertTrue(stats["is_circular_valid"])

    def test_append_and_circularity(self):
        self.cll.append(self.r1)
        self.cll.append(self.r2)
        self.cll.append(self.r3)

        self.assertEqual(len(self.cll), 3)
        self.assertFalse(self.cll.is_empty())

        # Test circularity: traverse 7 steps and check cycle
        n1 = self.cll.head
        self.assertEqual(n1.record.scan_id, "SCAN-001")
        n2 = self.cll.get_next(n1)
        self.assertEqual(n2.record.scan_id, "SCAN-002")
        n3 = self.cll.get_next(n2)
        self.assertEqual(n3.record.scan_id, "SCAN-003")
        n4 = self.cll.get_next(n3)
        self.assertEqual(n4.record.scan_id, "SCAN-001")  # Loops back to head!

        # Check get_prev
        self.assertEqual(self.cll.get_prev(n1).record.scan_id, "SCAN-003")
        self.assertEqual(self.cll.get_prev(n2).record.scan_id, "SCAN-001")
        self.assertEqual(self.cll.get_prev(n3).record.scan_id, "SCAN-002")

        stats = self.cll.get_stats()
        self.assertTrue(stats["is_circular_valid"])
        self.assertEqual(stats["head_id"], "SCAN-001")
        self.assertEqual(stats["tail_id"], "SCAN-003")

    def test_deletion_scenarios(self):
        # 1. Single node deletion
        single_list = CircularLinkedList()
        single_list.append(self.r1)
        self.assertTrue(single_list.delete_by_id("SCAN-001"))
        self.assertTrue(single_list.is_empty())
        self.assertEqual(len(single_list), 0)

        # 2. Deleting head from multi-node list
        self.cll.append(self.r1)
        self.cll.append(self.r2)
        self.cll.append(self.r3)
        self.assertTrue(self.cll.delete_by_id("SCAN-001"))
        self.assertEqual(len(self.cll), 2)
        self.assertEqual(self.cll.head.record.scan_id, "SCAN-002")
        self.assertEqual(self.cll.get_next(self.cll.head.next).record.scan_id, "SCAN-002")

        # 3. Deleting tail
        self.assertTrue(self.cll.delete_by_id("SCAN-003"))
        self.assertEqual(len(self.cll), 1)
        self.assertEqual(self.cll.head.record.scan_id, "SCAN-002")
        self.assertEqual(self.cll.head.next, self.cll.head)


class TestAVLBalancedTree(unittest.TestCase):
    """Test suite for CO3: AVL Balanced Binary Search Tree."""

    def setUp(self):
        self.tree = AVLTree()

    def test_rotations_ll_rr_lr_rl(self):
        # Test LL rotation: inserting decreasing keys (30, 20, 10)
        tree_ll = AVLTree()
        tree_ll.insert(TrafficRecord("S3", "", 0, 0, 0, 0, 30, "High", 30.0, ""))
        tree_ll.insert(TrafficRecord("S2", "", 0, 0, 0, 0, 20, "Medium", 20.0, ""))
        tree_ll.insert(TrafficRecord("S1", "", 0, 0, 0, 0, 10, "Low", 10.0, ""))
        stats_ll = tree_ll.get_tree_stats()
        self.assertTrue(stats_ll["is_strictly_balanced"])
        self.assertEqual(stats_ll["rotations"]["LL"], 1)
        self.assertEqual(tree_ll.root.record.scan_id, "S2")

        # Test RR rotation: inserting increasing keys (10, 20, 30)
        tree_rr = AVLTree()
        tree_rr.insert(TrafficRecord("S1", "", 0, 0, 0, 0, 10, "Low", 10.0, ""))
        tree_rr.insert(TrafficRecord("S2", "", 0, 0, 0, 0, 20, "Medium", 20.0, ""))
        tree_rr.insert(TrafficRecord("S3", "", 0, 0, 0, 0, 30, "High", 30.0, ""))
        stats_rr = tree_rr.get_tree_stats()
        self.assertTrue(stats_rr["is_strictly_balanced"])
        self.assertEqual(stats_rr["rotations"]["RR"], 1)
        self.assertEqual(tree_rr.root.record.scan_id, "S2")

        # Test LR rotation: keys (30, 10, 20)
        tree_lr = AVLTree()
        tree_lr.insert(TrafficRecord("S3", "", 0, 0, 0, 0, 30, "High", 30.0, ""))
        tree_lr.insert(TrafficRecord("S1", "", 0, 0, 0, 0, 10, "Low", 10.0, ""))
        tree_lr.insert(TrafficRecord("S2", "", 0, 0, 0, 0, 20, "Medium", 20.0, ""))
        stats_lr = tree_lr.get_tree_stats()
        self.assertTrue(stats_lr["is_strictly_balanced"])
        self.assertEqual(stats_lr["rotations"]["LR"], 1)
        self.assertEqual(tree_lr.root.record.scan_id, "S2")

        # Test RL rotation: keys (10, 30, 20)
        tree_rl = AVLTree()
        tree_rl.insert(TrafficRecord("S1", "", 0, 0, 0, 0, 10, "Low", 10.0, ""))
        tree_rl.insert(TrafficRecord("S3", "", 0, 0, 0, 0, 30, "High", 30.0, ""))
        tree_rl.insert(TrafficRecord("S2", "", 0, 0, 0, 0, 20, "Medium", 20.0, ""))
        stats_rl = tree_rl.get_tree_stats()
        self.assertTrue(stats_rl["is_strictly_balanced"])
        self.assertEqual(stats_rl["rotations"]["RL"], 1)
        self.assertEqual(tree_rl.root.record.scan_id, "S2")

    def test_inorder_and_search(self):
        records = [
            TrafficRecord("S05", "", 0, 0, 0, 0, 15, "Medium", 15.0, ""),
            TrafficRecord("S02", "", 0, 0, 0, 0, 4, "Low", 4.0, ""),
            TrafficRecord("S08", "", 0, 0, 0, 0, 22, "High", 22.0, ""),
            TrafficRecord("S01", "", 0, 0, 0, 0, 1, "Low", 1.0, ""),
            TrafficRecord("S06", "", 0, 0, 0, 0, 18, "High", 18.0, ""),
        ]
        for r in records:
            self.tree.insert(r)

        # In-order traversal should return strictly sorted by score
        sorted_records = self.tree.in_order_traversal()
        scores = [r.congestion_score for r in sorted_records]
        self.assertEqual(scores, [1.0, 4.0, 15.0, 18.0, 22.0])

        # Search by scan ID
        found = self.tree.search_by_id("S08")
        self.assertNotNull = found is not None
        self.assertEqual(found.congestion_score, 22.0)
        self.assertIsNone(self.tree.search_by_id("NONEXISTENT"))

        # Search by score range
        range_matches = self.tree.search_by_score_range(3.0, 19.0)
        range_scores = [r.congestion_score for r in range_matches]
        self.assertEqual(range_scores, [4.0, 15.0, 18.0])


class TestRoadGraphAndDFS(unittest.TestCase):
    """Test suite for CO5: Road Network Graph & DFS Route Cost Evaluator."""

    def setUp(self):
        self.graph = create_sample_road_network()

    def test_dfs_route_discovery_and_cycle_prevention(self):
        start = "J1_GATEWAY"
        dest = "J7_TERMINAL"
        routes = self.graph.find_routes_dfs(start, dest)

        # There should be multiple paths connecting J1 to J7
        self.assertGreater(len(routes), 1)

        for r in routes:
            path = r["path"]
            self.assertEqual(path[0], start)
            self.assertEqual(path[-1], dest)
            # Cycle prevention verification: all vertices in path must be unique
            self.assertEqual(len(path), len(set(path)))
            self.assertGreater(r["total_distance_km"], 0.0)
            self.assertGreater(r["total_congestion_cost"], 0.0)

    def test_route_evaluation_and_cost_weighting(self):
        start = "J1_GATEWAY"
        dest = "J7_TERMINAL"
        routes = self.graph.find_routes_dfs(start, dest)

        # Prioritize distance (w_dist=0.9, w_cong=0.1)
        best_dist, _ = self.graph.evaluate_routes(routes, weight_distance=0.9, weight_congestion=0.1)
        self.assertIsNotNone(best_dist)

        # Prioritize congestion avoidance (w_dist=0.1, w_cong=0.9)
        best_cong, _ = self.graph.evaluate_routes(routes, weight_distance=0.1, weight_congestion=0.9)
        self.assertIsNotNone(best_cong)

        # Ensure composite cost is correctly computed
        expected_cost = round((0.9 * best_dist["total_distance_km"]) + (0.1 * best_dist["total_congestion_cost"]), 3)
        self.assertEqual(best_dist["composite_cost"], expected_cost)

    def test_no_route_handling(self):
        # Add an isolated vertex
        self.graph.add_intersection("J_ISOLATED", "Isolated Island", "Sector 99", (0, 0))
        routes = self.graph.find_routes_dfs("J1_GATEWAY", "J_ISOLATED")
        self.assertEqual(len(routes), 0)
        best, ranked = self.graph.evaluate_routes(routes)
        self.assertIsNone(best)
        self.assertEqual(len(ranked), 0)


if __name__ == "__main__":
    unittest.main()
