#!/usr/bin/env python3
"""
Analyze PANGRAM results to see distribution of ai_likelihood scores.
Groups results into bins of 0.05 increments.
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

OUTPUT_JSON = Path("pangram_abstracts_results.json")


def load_results() -> List[Dict]:
    """Load results from JSON file."""
    if not OUTPUT_JSON.exists():
        print(f"Error: {OUTPUT_JSON} not found")
        return []
    
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return list(data.values())
            else:
                return []
    except Exception as e:
        print(f"Error loading results: {e}")
        return []


def get_bin_label(lower: float, upper: float) -> str:
    """Get label for a bin range."""
    return f"{lower:.2f}-{upper:.2f}"


def analyze_results(results: List[Dict], bin_size: float = 0.05) -> Dict:
    """Analyze results and group by ai_likelihood bins."""
    # Initialize bins
    bins = defaultdict(int)
    bins_by_domain = defaultdict(lambda: defaultdict(int))
    errors = 0
    total = len(results)
    
    for result in results:
        # Check for errors
        if "error" in result:
            errors += 1
            continue
        
        # Get ai_likelihood
        ai_likelihood = result.get("ai_likelihood")
        if ai_likelihood is None:
            errors += 1
            continue
        
        # Determine which bin this falls into
        bin_index = int(ai_likelihood / bin_size)
        lower = bin_index * bin_size
        upper = (bin_index + 1) * bin_size
        
        # Handle edge case: exactly 1.0 goes into last bin
        if ai_likelihood >= 1.0:
            lower = 0.95
            upper = 1.0
            bin_index = 19
        
        bin_label = get_bin_label(lower, upper)
        bins[bin_label] += 1
        
        # Also track by domain
        domain = result.get("domain", "unknown")
        bins_by_domain[domain][bin_label] += 1
    
    return {
        "bins": dict(bins),
        "bins_by_domain": {k: dict(v) for k, v in bins_by_domain.items()},
        "errors": errors,
        "total": total,
        "successful": total - errors
    }


def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    bins = analysis["bins"]
    bins_by_domain = analysis["bins_by_domain"]
    errors = analysis["errors"]
    total = analysis["total"]
    successful = analysis["successful"]
    
    print("=" * 70)
    print("PANGRAM Results Analysis - AI Likelihood Distribution")
    print("=" * 70)
    print(f"\nTotal results: {total}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}\n")
    
    # Print overall distribution
    print("-" * 70)
    print("Overall Distribution (by 0.05 increments)")
    print("-" * 70)
    print(f"{'Range':<15} {'Count':<10} {'Percentage':<15} {'Bar'}")
    print("-" * 70)
    
    # Sort bins by lower bound
    sorted_bins = sorted(bins.items(), key=lambda x: float(x[0].split('-')[0]))
    
    for bin_label, count in sorted_bins:
        percentage = (count / successful * 100) if successful > 0 else 0
        bar_length = int(percentage / 2)  # Scale bar to fit in terminal
        bar = "█" * bar_length
        print(f"{bin_label:<15} {count:<10} {percentage:>6.2f}%{'':<8} {bar}")
    
    # Print by domain
    if bins_by_domain:
        print("\n" + "-" * 70)
        print("Distribution by Domain")
        print("-" * 70)
        
        for domain in sorted(bins_by_domain.keys()):
            domain_bins = bins_by_domain[domain]
            domain_total = sum(domain_bins.values())
            
            print(f"\n{domain.upper()} (Total: {domain_total})")
            print(f"{'Range':<15} {'Count':<10} {'Percentage':<15} {'Bar'}")
            print("-" * 70)
            
            sorted_domain_bins = sorted(domain_bins.items(), key=lambda x: float(x[0].split('-')[0]))
            
            for bin_label, count in sorted_domain_bins:
                percentage = (count / domain_total * 100) if domain_total > 0 else 0
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                print(f"{bin_label:<15} {count:<10} {percentage:>6.2f}%{'':<8} {bar}")
    
    # Summary statistics
    print("\n" + "-" * 70)
    print("Summary Statistics")
    print("-" * 70)
    
    # Calculate some basic stats
    all_likelihoods = []
    for result in load_results():
        if "error" not in result and "ai_likelihood" in result:
            all_likelihoods.append(result["ai_likelihood"])
    
    if all_likelihoods:
        all_likelihoods.sort()
        print(f"Min ai_likelihood: {min(all_likelihoods):.6f}")
        print(f"Max ai_likelihood: {max(all_likelihoods):.6f}")
        print(f"Mean ai_likelihood: {sum(all_likelihoods) / len(all_likelihoods):.6f}")
        median_idx = len(all_likelihoods) // 2
        median = all_likelihoods[median_idx] if len(all_likelihoods) % 2 == 1 else (all_likelihoods[median_idx - 1] + all_likelihoods[median_idx]) / 2
        print(f"Median ai_likelihood: {median:.6f}")
        
        # Count in key ranges
        very_low = sum(1 for l in all_likelihoods if 0 <= l < 0.05)
        low = sum(1 for l in all_likelihoods if 0.05 <= l < 0.25)
        medium = sum(1 for l in all_likelihoods if 0.25 <= l < 0.75)
        high = sum(1 for l in all_likelihoods if 0.75 <= l <= 1.0)
        
        print(f"\nKey Ranges:")
        print(f"  0.00-0.05 (Very Unlikely AI): {very_low} ({very_low/len(all_likelihoods)*100:.1f}%)")
        print(f"  0.05-0.25 (Unlikely AI): {low} ({low/len(all_likelihoods)*100:.1f}%)")
        print(f"  0.25-0.75 (Uncertain): {medium} ({medium/len(all_likelihoods)*100:.1f}%)")
        print(f"  0.75-1.00 (Likely AI): {high} ({high/len(all_likelihoods)*100:.1f}%)")
    
    print("=" * 70)


def main():
    """Main function."""
    print("Loading PANGRAM results...\n")
    results = load_results()
    
    if not results:
        print("No results found to analyze.")
        return
    
    print(f"Loaded {len(results)} results\n")
    
    # Analyze results
    analysis = analyze_results(results, bin_size=0.05)
    
    # Print analysis
    print_analysis(analysis)


if __name__ == "__main__":
    main()
