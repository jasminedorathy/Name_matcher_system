import json
import os
from difflib import SequenceMatcher
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NameMatcher:
    def __init__(self, data_file: str = "data/names.json"):
        self.data_file = data_file
        self.names = self.load_names()
        self.vectorizer = None
        self.tfidf_matrix = None
        self._setup_tfidf()
    
    def load_names(self) -> List[str]:
        """Load names from JSON file"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('names', [])
        except FileNotFoundError:
            print(f"Warning: {self.data_file} not found. Using default names.")
            return self._get_default_names()
    
    def _get_default_names(self) -> List[str]:
        """Return default list of similar names"""
        return [
            "Geetha", "Gita", "Gitu", "Geeta", "Githa",
            "John", "Jon", "Johny", "Jonathan", "Jonathon",
            "Michael", "Mike", "Micheal", "Mikael", "Mikhail",
            "Sarah", "Sara", "Saira", "Zara", "Sahra",
            "Robert", "Rob", "Bob", "Roberto", "Robby",
            "Jennifer", "Jenny", "Jenn", "Jenifer", "Jenna",
            "Christopher", "Chris", "Kristopher", "Topher", "Cristobal",
            "Elizabeth", "Liz", "Beth", "Eliza", "Liza",
            "William", "Will", "Bill", "Billy", "Willy",
            "Katherine", "Kate", "Katie", "Catherine", "Kat"
        ]
    
    def _setup_tfidf(self):
        """Setup TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.names)
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def normalized_similarity(self, s1: str, s2: str) -> float:
        """Calculate normalized similarity score (0-1)"""
        distance = self.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - (distance / max_len)
    
    def sequence_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity using SequenceMatcher"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def tfidf_similarity(self, query: str) -> List[Tuple[str, float]]:
        """Calculate similarity using TF-IDF and cosine similarity"""
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            results = list(zip(self.names, similarities))
            return sorted(results, key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"TF-IDF similarity failed: {e}")
            return []
    
    def combined_similarity(self, query: str) -> List[Tuple[str, float]]:
        """Combine multiple similarity methods for better accuracy"""
        results = []
        
        for name in self.names:
            # Weighted combination of different similarity measures
            seq_score = self.sequence_similarity(query, name)
            norm_score = self.normalized_similarity(query, name)
            
            # Combined score (you can adjust weights)
            combined_score = (seq_score * 0.6) + (norm_score * 0.4)
            results.append((name, combined_score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def find_matches(self, query: str, method: str = "combined", top_k: int = 5) -> Dict:
        """Find matching names based on query"""
        if method == "tfidf":
            matches = self.tfidf_similarity(query)
        elif method == "sequence":
            matches = [(name, self.sequence_similarity(query, name)) for name in self.names]
            matches = sorted(matches, key=lambda x: x[1], reverse=True)
        elif method == "levenshtein":
            matches = [(name, self.normalized_similarity(query, name)) for name in self.names]
            matches = sorted(matches, key=lambda x: x[1], reverse=True)
        else:  # combined
            matches = self.combined_similarity(query)
        
        # Filter out zero-similarity matches for cleaner output
        matches = [(name, score) for name, score in matches if score > 0]
        
        return {
            "best_match": matches[0] if matches else (None, 0.0),
            "all_matches": matches[:top_k],
            "query": query,
            "method": method
        }
    
    def add_name(self, name: str):
        """Add a new name to the dataset"""
        if name not in self.names:
            self.names.append(name)
            self._save_names()
            self._setup_tfidf()  # Rebuild TF-IDF matrix
    
    def _save_names(self):
        """Save names to JSON file"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        data = {"names": self.names}
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

# Singleton instance
name_matcher = None

def get_name_matcher():
    global name_matcher
    if name_matcher is None:
        name_matcher = NameMatcher()
    return name_matcher