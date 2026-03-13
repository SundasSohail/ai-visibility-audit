"""
🔎 Google AI Search: Simulating Query Fan-Out Visibility
AI Visibility Audit Tool - Local Version

This script analyzes how well a given URL's content might be visible to
Google's AI search mechanisms by simulating its "query fan-out" process.

Requirements:
- Google Gemini API key
- OpenAI API key (for embeddings)

Author: Andrea Volpini & AI Assistant
Last updated: June 9th, 2025
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse
import re
import json
from typing import Dict, List, Tuple

import requests
import numpy as np
import dspy
from google import genai
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# ============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ============================================================================

def initialize_api_keys():
    """Initialize and validate API keys from environment variables."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in .env file or environment variables."
        )
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in .env file or environment variables."
        )
    
    return google_api_key, openai_api_key


def setup_dspy_and_gemini():
    """Configure DSPy and Gemini for the audit."""
    google_api_key, openai_api_key = initialize_api_keys()
    
    # Initialize Gemini client
    genai.configure(api_key=google_api_key)
    gemini_client = genai.Client()
    
    # Configure DSPy with Gemini
    gemini_lm = dspy.LM(
        model="gemini-2.5-flash",
        api_key=google_api_key,
        max_tokens=2048,
    )
    dspy.settings.configure(lm=gemini_lm)
    
    print("✅ Gemini API initialized")
    print("✅ DSPy configured with Gemini (gemini-2.5-flash)")
    
    return gemini_client, gemini_lm


# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def get_url_insights_and_content_via_gemini(url_to_process: str, gemini_client) -> Dict:
    """
    Uses the Gemini client with url_context tool to get the main entity
    and grounded content chunks from a URL.
    
    Args:
        url_to_process: The URL to analyze
        gemini_client: Initialized Gemini client
        
    Returns:
        Dictionary with entity info and content chunks
    """
    try:
        # Fetch content from URL
        response = requests.get(url_to_process, timeout=10)
        response.raise_for_status()
        content = response.text
        
        # Use Gemini to extract entity and summarize content
        model = gemini_client.models.get("models/gemini-2.5-flash")
        
        prompt = f"""Analyze this webpage content and provide:
1. The main entity/topic of this page (one concise phrase)
2. Key content chunks (3-5 main topics covered)

URL: {url_to_process}
Content excerpt: {content[:2000]}

Respond in JSON format:
{{
    "entity": "main topic here",
    "content_chunks": ["chunk1", "chunk2", "chunk3"]
}}
"""
        
        response = model.generate_content(prompt)
        
        # Parse JSON response
        result_text = response.text
        try:
            # Extract JSON from response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            json_str = result_text[json_start:json_end]
            result = json.loads(json_str)
        except:
            # Fallback if JSON parsing fails
            result = {
                "entity": "Unable to extract",
                "content_chunks": ["Content analysis failed"],
                "raw_response": result_text
            }
        
        return result
        
    except Exception as e:
        print(f"⚠️  Error fetching URL insights: {e}")
        return {
            "entity": "Error",
            "content_chunks": [str(e)],
            "error": True
        }


# ============================================================================
# 3. DSPY SIGNATURES AND MODULES
# ============================================================================

class GenerateSyntheticQueriesCoT(dspy.Signature):
    """
    Given an entity (typically the main topic of a webpage), the current date,
    and a desired number of sub-queries, generate synthetic queries that Google's
    AI search might use internally to understand the entity comprehensively.
    
    Reason about key facets, common questions, and related concepts.
    """
    
    entity = dspy.InputField(
        desc="The main entity/topic to generate queries for"
    )
    current_date = dspy.InputField(
        desc="The current date for context"
    )
    num_queries = dspy.InputField(
        desc="The number of synthetic queries to generate"
    )
    reasoning = dspy.OutputField(
        desc="Brief reasoning about key facets and related concepts"
    )
    synthetic_queries = dspy.OutputField(
        desc="A list of synthetic sub-queries, one per line"
    )


class FanOutQueryChecker(dspy.ChainOfThought):
    """Check if given content effectively answers a specific synthetic query."""
    
    query = dspy.InputField(
        desc="The synthetic query to check"
    )
    content = dspy.InputField(
        desc="The webpage content to check against"
    )
    reasoning = dspy.OutputField(
        desc="Reasoning about content relevance"
    )
    relevance_score = dspy.OutputField(
        desc="Score from 0 to 1 indicating how well content answers the query"
    )


class SyntheticQueryGenerator(dspy.Module):
    """Module for generating synthetic queries using Chain of Thought."""
    
    def __init__(self):
        super().__init__()
        self.generate_queries = dspy.ChainOfThought(GenerateSyntheticQueriesCoT)
    
    def forward(self, entity: str, current_date: str, num_queries: int = 10):
        """Generate synthetic queries for an entity."""
        result = self.generate_queries(
            entity=entity,
            current_date=current_date,
            num_queries=str(num_queries)
        )
        
        # Parse queries from output
        queries_text = result.synthetic_queries
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        
        return {
            "reasoning": result.reasoning,
            "queries": queries[:num_queries]
        }


class ContentCoverageChecker(dspy.Module):
    """Module for checking content coverage against synthetic queries."""
    
    def __init__(self):
        super().__init__()
        self.check_relevance = dspy.ChainOfThought(FanOutQueryChecker)
    
    def forward(self, query: str, content: str):
        """Check if content covers a query."""
        result = self.check_relevance(query=query, content=content)
        
        try:
            score = float(result.relevance_score)
        except:
            score = 0.5  # Default score if parsing fails
        
        return {
            "reasoning": result.reasoning,
            "relevance_score": min(max(score, 0), 1)  # Clamp between 0-1
        }


# ============================================================================
# 4. MAIN AUDIT FUNCTION
# ============================================================================

def run_ai_visibility_audit(
    url: str,
    num_synthetic_queries: int = 10,
    coverage_threshold: float = 0.55,
    gemini_client=None
) -> Dict:
    """
    Run the complete AI visibility audit for a URL.
    
    Args:
        url: URL to audit
        num_synthetic_queries: Number of synthetic queries to generate
        coverage_threshold: Minimum relevance score to consider content as covering a query
        gemini_client: Initialized Gemini client
        
    Returns:
        Dictionary with audit results
    """
    
    print(f"\n{'='*70}")
    print(f"🔍 Starting AI Visibility Audit")
    print(f"{'='*70}")
    print(f"URL: {url}")
    print(f"Synthetic Queries: {num_synthetic_queries}")
    print(f"Coverage Threshold: {coverage_threshold}")
    
    # Step 1: Get URL insights
    print("\n[Step 1] Extracting URL entity and content...")
    url_insights = get_url_insights_and_content_via_gemini(url, gemini_client)
    
    if "error" in url_insights:
        print(f"❌ Failed to extract URL insights: {url_insights.get('error')}")
        return {"error": True, "message": "Failed to extract URL insights"}
    
    entity = url_insights.get("entity", "Unknown")
    print(f"✅ Entity identified: {entity}")
    
    # Fetch actual content from URL for coverage checking
    print("\n[Step 2] Fetching URL content...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        url_content = response.text[:5000]  # Use first 5000 chars for efficiency
        print(f"✅ Content retrieved ({len(url_content)} characters)")
    except Exception as e:
        print(f"⚠️  Error fetching URL content: {e}")
        url_content = ""
    
    # Step 2: Generate synthetic queries
    print("\n[Step 3] Generating synthetic queries...")
    query_generator = SyntheticQueryGenerator()
    
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    query_result = query_generator.forward(
        entity=entity,
        current_date=current_date,
        num_queries=num_synthetic_queries
    )
    
    synthetic_queries = query_result["queries"]
    print(f"✅ Generated {len(synthetic_queries)} synthetic queries:")
    for i, q in enumerate(synthetic_queries, 1):
        print(f"   {i}. {q}")
    
    # Step 3: Check content coverage
    print("\n[Step 4] Checking content coverage...")
    coverage_checker = ContentCoverageChecker()
    
    coverage_results = []
    covered_count = 0
    
    for i, query in enumerate(synthetic_queries, 1):
        print(f"\n   Query {i}/{len(synthetic_queries)}: {query}")
        
        check_result = coverage_checker.forward(
            query=query,
            content=url_content
        )
        
        relevance_score = check_result["relevance_score"]
        is_covered = relevance_score >= coverage_threshold
        
        if is_covered:
            covered_count += 1
            print(f"   ✅ Covered (Score: {relevance_score:.2f})")
        else:
            print(f"   ❌ Not covered (Score: {relevance_score:.2f})")
        
        coverage_results.append({
            "query": query,
            "relevance_score": relevance_score,
            "is_covered": is_covered,
            "reasoning": check_result["reasoning"]
        })
    
    # Calculate overall coverage
    coverage_percentage = (covered_count / len(synthetic_queries)) * 100
    
    # Final results
    print(f"\n{'='*70}")
    print(f"📊 AUDIT RESULTS")
    print(f"{'='*70}")
    print(f"Entity: {entity}")
    print(f"Total Queries: {len(synthetic_queries)}")
    print(f"Covered Queries: {covered_count}")
    print(f"Coverage Percentage: {coverage_percentage:.1f}%")
    print(f"Coverage Threshold: {coverage_threshold:.0%}")
    
    if coverage_percentage >= (coverage_threshold * 100):
        print(f"✅ PASSED: URL has good AI visibility")
    else:
        print(f"❌ FAILED: URL lacks sufficient AI visibility")
    
    print(f"{'='*70}\n")
    
    return {
        "url": url,
        "entity": entity,
        "num_synthetic_queries": len(synthetic_queries),
        "covered_count": covered_count,
        "coverage_percentage": coverage_percentage,
        "coverage_threshold": coverage_threshold,
        "passed": coverage_percentage >= (coverage_threshold * 100),
        "query_results": coverage_results
    }


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for the audit tool."""
    print("🔎 Google AI Search: Simulating Query Fan-Out Visibility\n")
    
    # Setup
    print("Initializing...")
    gemini_client, gemini_lm = setup_dspy_and_gemini()
    
    # Get user input
    print("\n" + "="*70)
    url_to_audit = input("Enter the URL to analyze: ").strip()
    
    if not url_to_audit:
        print("Error: URL is required")
        return
    
    # Validate URL
    try:
        result = urlparse(url_to_audit)
        if not all([result.scheme, result.netloc]):
            raise ValueError("Invalid URL format")
    except:
        print("Error: Invalid URL format. Please include http:// or https://")
        return
    
    num_queries_input = input("Number of synthetic queries (3-10) [default: 10]: ").strip()
    try:
        num_queries = int(num_queries_input) if num_queries_input else 10
        num_queries = max(3, min(10, num_queries))
    except ValueError:
        num_queries = 10
    
    threshold_input = input("Coverage threshold (0.5-0.9) [default: 0.55]: ").strip()
    try:
        threshold = float(threshold_input) if threshold_input else 0.55
        threshold = max(0.5, min(0.9, threshold))
    except ValueError:
        threshold = 0.55
    
    # Run audit
    results = run_ai_visibility_audit(
        url=url_to_audit,
        num_synthetic_queries=num_queries,
        coverage_threshold=threshold,
        gemini_client=gemini_client
    )
    
    # Save results
    output_file = "audit_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAudit cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
