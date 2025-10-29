#!/usr/bin/env python3
"""
Name Matching System - Main Application (CLI Version)
"""

import json
from name_matcher import get_name_matcher

def display_results(results):
    """Display matching results in a formatted way"""
    print("\n" + "="*50)
    print(f"QUERY: '{results['query']}'")
    print(f"METHOD: {results['method']}")
    print("="*50)
    
    best_match, best_score = results['best_match']
    print(f"🏆 BEST MATCH: {best_match} (Score: {best_score:.4f})")
    
    print("\n📊 TOP MATCHES:")
    print("-" * 30)
    for i, (name, score) in enumerate(results['all_matches'], 1):
        print(f"{i:2d}. {name:<15} → Score: {score:.4f}")
    
    print("="*50)

def main():
    """Main application loop"""
    matcher = get_name_matcher()
    
    print("🔍 NAME MATCHING SYSTEM")
    print("Available similarity methods:")
    print("1. combined (default) - Best overall accuracy")
    print("2. sequence - Based on sequence matching")
    print("3. levenshtein - Based on edit distance")
    print("4. tfidf - Based on character n-grams")
    
    while True:
        print("\n" + "━" * 40)
        query = input("\nEnter a name to search (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not query:
            print("Please enter a valid name.")
            continue
        
        method = input("Choose method [combined/sequence/levenshtein/tfidf] (default: combined): ").strip().lower()
        if method not in ['combined', 'sequence', 'levenshtein', 'tfidf']:
            method = 'combined'
            print("Using default method: combined")
        
        try:
            results = matcher.find_matches(query, method=method, top_k=10)
            display_results(results)
            
            # Option to add new name
            add_new = input("\nAdd this query to dataset? (y/N): ").strip().lower()
            if add_new == 'y':
                matcher.add_name(query)
                print(f"✅ '{query}' added to dataset!")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()