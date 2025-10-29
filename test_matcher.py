#!/usr/bin/env python3
"""
Test suite for Name Matching System
"""

import unittest
from name_matcher import NameMatcher

class TestNameMatcher(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixture"""
        self.matcher = NameMatcher("data/names.json")
    
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation"""
        self.assertEqual(self.matcher.levenshtein_distance("cat", "cat"), 0)
        self.assertEqual(self.matcher.levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(self.matcher.levenshtein_distance("", "test"), 4)
    
    def test_normalized_similarity(self):
        """Test normalized similarity score"""
        self.assertAlmostEqual(self.matcher.normalized_similarity("cat", "cat"), 1.0)
        self.assertGreater(self.matcher.normalized_similarity("cat", "bat"), 0.5)
        self.assertLess(self.matcher.normalized_similarity("cat", "dog"), 0.5)
    
    def test_sequence_similarity(self):
        """Test sequence similarity"""
        self.assertEqual(self.matcher.sequence_similarity("test", "test"), 1.0)
        self.assertGreater(self.matcher.sequence_similarity("john", "jon"), 0.5)
    
    def test_find_matches_combined(self):
        """Test combined matching method"""
        results = self.matcher.find_matches("Geetha", method="combined", top_k=3)
        
        self.assertIsNotNone(results["best_match"])
        self.assertEqual(len(results["all_matches"]), 3)
        
        best_match, best_score = results["best_match"]
        self.assertEqual(best_match, "Geetha")
        self.assertAlmostEqual(best_score, 1.0, places=2)
    
    def test_find_matches_tfidf(self):
        """Test TF-IDF matching method"""
        results = self.matcher.find_matches("John", method="tfidf", top_k=3)
        
        self.assertIsNotNone(results["best_match"])
        self.assertEqual(len(results["all_matches"]), 3)
        
        # Should find similar names
        match_names = [name for name, score in results["all_matches"]]
        self.assertTrue(any("John" in name or "Jon" in name for name in match_names))
    
    def test_add_name(self):
        """Test adding new name to dataset"""
        initial_count = len(self.matcher.names)
        self.matcher.add_name("TestUniqueName123")
        
        self.assertEqual(len(self.matcher.names), initial_count + 1)
        self.assertIn("TestUniqueName123", self.matcher.names)

def run_tests():
    """Run all tests"""
    print("Running Name Matcher Tests...")
    unittest.main(verbosity=2, exit=False)

if __name__ == "__main__":
    run_tests()