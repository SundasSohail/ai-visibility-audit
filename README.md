# 🔎 Google AI Search: Simulating Query Fan-Out Visibility

An AI Visibility Audit Tool that simulates how well a URL's content would be visible to Google's AI search mechanisms.

## Overview

This tool analyzes a given URL by:

1. **Entity Identification**: Determines the main entity/topic of a URL
2. **Synthetic Query Generation**: Generates sub-queries that Google's AI might use internally
3. **Content Coverage Assessment**: Checks if the URL's content effectively answers these synthetic queries

## Requirements

- Python 3.8+
- Google Generative AI (Gemini) API key
- OpenAI API key (for embeddings)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

```
GOOGLE_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**How to get API keys:**
- **Google Gemini API**: https://ai.google.dev/
- **OpenAI API**: https://platform.openai.com/api-keys

## Usage

Run the audit tool:

```bash
python ai_visibility_audit.py
```

The script will prompt you for:
- **URL**: The webpage URL to analyze
- **Number of synthetic queries** (3-10, default: 10)
- **Coverage threshold** (0.5-0.9, default: 0.55)

### Example

```bash
$ python ai_visibility_audit.py
🔎 Google AI Search: Simulating Query Fan-Out Visibility

Initializing...
✅ Gemini API initialized
✅ DSPy configured with Gemini (gemini-2.5-flash)

======================================================================
Enter the URL to analyze: https://example.com/article
Number of synthetic queries (3-10) [default: 10]: 10
Coverage threshold (0.5-0.9) [default: 0.55]: 0.55

======================================================================
🔍 Starting AI Visibility Audit
======================================================================
...
```

## Output

The tool generates an `audit_results.json` file with detailed results:

```json
{
  "url": "https://example.com/article",
  "entity": "Main topic here",
  "num_synthetic_queries": 10,
  "covered_count": 8,
  "coverage_percentage": 80.0,
  "coverage_threshold": 0.55,
  "passed": true,
  "query_results": [...]
}
```

## Key Features

- **DSPy Integration**: Uses DSPy for task-oriented module building
- **Chain of Thought Reasoning**: LLM provides reasoning for each decision
- **Google Gemini LLM**: Leverages Gemini 2.5 Flash for fast, efficient analysis
- **Customizable Parameters**: Adjust query count and coverage threshold
- **Detailed Reports**: Get comprehensive results for each query

## Architecture

- `ai_visibility_audit.py`: Main audit script
- `requirements.txt`: Python dependencies
- `.env`: API keys configuration (create from .env.example)

## Concept & Patent References

Based on:
- US Patent 2024/0289407 A1: Patent mechanisms for AI search
- WO2024064249A1: Systems and methods for prompt-based query generation for diverse retrieval
- Analysis by Michael King (iPullRank) on AI Mode query fan-out processes

Created by WordLift  
Author: Andrea Volpini & AI Assistant  
Last updated: June 9th, 2025

## Troubleshooting

### Missing API Keys
If you get an error about missing API keys:
1. Ensure `.env` file exists in the project root
2. Check that both `GOOGLE_API_KEY` and `OPENAI_API_KEY` are set
3. Verify the keys are valid and not expired

### Connection Errors
If the tool can't reach a URL:
- Verify the URL is accessible and includes `http://` or `https://`
- Check your internet connection
- Ensure the website allows automated access

### API Rate Limits
If you encounter rate limiting:
- Wait a few moments before running another audit
- Consider using a higher tier API plan for increased limits

## License

This tool is provided for research and analysis purposes.
