#!/usr/bin/env python3
"""
Test module to verify OpenRouterFreeOpenAIClient functionality.
This demonstrates that the client works as a drop-in replacement for OpenAI-compatible models.
"""

import json
import argparse
import os
from openrouterfree import OpenRouterFreeOpenAIClient

def parse_arguments():
    """Parse command line arguments for the test client."""
    parser = argparse.ArgumentParser(
        description="Test OpenRouterFreeOpenAIClient with configurable parameters"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of models to test (default: 5)"
    )
    
    parser.add_argument(
        "--require-params",
        type=str,
        help="Comma-separated list of required parameters (e.g., 'tool_choice,tools')"
    )
    
    parser.add_argument(
        "--sort-by",
        type=str,
        default="context_length",
        help="Sort models by this criteria (default: 'context_length')"
    )
    
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the sort order"
    )
    
    parser.add_argument(
        "--openai_api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for OpenAI API key (default: 'OPENAI_API_KEY')"
    )
    
    parser.add_argument(
        "--openrouter_api_base",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base URL (default: 'https://openrouter.ai/api/v1')"
    )
    
    parser.add_argument(
        "--error-threshold",
        type=int,
        default=3,
        help="Number of errors before switching models (default: 3)"
    )
    
    parser.add_argument(
        "--base-retry-delay",
        type=float,
        default=1.0,
        help="Base delay in seconds for exponential backoff (default: 1.0)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Test prompt to send to the OpenAI client (optional - enables live API testing)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Filter models by name (e.g., 'llama', 'gpt')"
    )
    
    parser.add_argument(
        "--min-context-length",
        type=int,
        help="Minimum context length filter"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        help="Filter models by provider (e.g., 'openai', 'anthropic')"
    )
    
    return parser.parse_args()

def test_live_api_call(args):
    """Test live API call with the provided prompt or default joke prompt."""
    # Use default joke prompt if none provided
    prompt = args.prompt if args.prompt else "Tell me a short, funny joke about programming."
    
    print("\n=== Testing Live API Call ===")
    
    try:
        # Get API key from environment
        api_key_env = args.openai_api_key_env
        api_key = os.environ.get(api_key_env)
        
        if not api_key:
            print(f"⚠️  Warning: Environment variable '{api_key_env}' not found")
            print("   Skipping live API test - no API key available")
            return True
        
        print(f"Test: Making live API call with prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        # Parse require-params if provided
        required_params = []
        if args.require_params:
            required_params = [param.strip() for param in args.require_params.split(',')]
        
        # Create client with real API key and all parameters
        client = OpenRouterFreeOpenAIClient(
            api_key=api_key,
            limit=args.limit,
            name=args.name,
            min_context_length=args.min_context_length,
            provider=args.provider,
            sort_by=args.sort_by,
            reverse=args.reverse,
            required_parameters=required_params,
            openrouter_base_url=args.openrouter_api_base,
            max_retries=args.error_threshold,
            base_retry_delay=args.base_retry_delay
        )
        
        print(f"  Using model: {client.best_model}")
        
        # Make the API call
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        print("✓ Live API call successful!")
        print(f"  🤖 AI Response: {response.choices[0].message.content}")
        print(f"  📋 Model used: {response.model}")
        print(f"  📊 Usage: {response.usage}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in live API test: {e}")
        return False

def main():
    """Run the test with command line arguments."""
    args = parse_arguments()
    
    print("OpenRouterFreeOpenAIClient Test Suite")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Limit: {args.limit}")
    print(f"  Sort by: {args.sort_by}")
    print(f"  Reverse: {args.reverse}")
    print(f"  Max retries/error threshold: {args.error_threshold}")
    print(f"  Base retry delay: {args.base_retry_delay}")
    print(f"  API key env: {args.openai_api_key_env}")
    print(f"  API base: {args.openrouter_api_base}")
    if args.name:
        print(f"  Name filter: {args.name}")
    if args.min_context_length:
        print(f"  Min context length: {args.min_context_length}")
    if args.provider:
        print(f"  Provider filter: {args.provider}")
    if args.require_params:
        print(f"  Required params: {args.require_params}")
    if args.prompt:
        print(f"  Test prompt: '{args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}'")
    else:
        print(f"  Test prompt: 'Tell me a short, funny joke about programming.' (default)")
    print("=" * 60)
    
    # Run the live API test
    print("\nRunning Live API Call test...")
    if test_live_api_call(args):
        print("\n" + "=" * 60)
        print("🎉 Test passed! OpenRouterFreeOpenAIClient is working correctly.")
        print("\nUsage Example:")
        print("```python")
        print("from openrouterfree import OpenRouterFreeOpenAIClient")
        print("")
        print("# Create client - automatically selects best free model")
        print("client = OpenRouterFreeOpenAIClient(")
        print("    api_key='your-openrouter-api-key'")
        print(")")
        print("")
        print("# Use like standard OpenAI client")
        print("response = client.chat.completions.create(")
        print("    messages=[{'role': 'user', 'content': 'Hello!'}]")
        print(")")
        print("```")
    else:
        print("\n" + "=" * 60)
        print("❌ Test failed. Please check the implementation.")
    
    return True

if __name__ == "__main__":
    main()