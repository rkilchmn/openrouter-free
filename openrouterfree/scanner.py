import requests
import json
import argparse
from typing import List, Dict, Any, Optional

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

def get_free_models(exclude_routers=True, base_url=OPENROUTER_DEFAULT_BASE_URL, api_key=None):
    """
    Fetches the list of all models from the OpenRouter API and returns the free ones.

    Args:
        exclude_routers (bool): If True, excludes router models.
        base_url (str): Base URL for the OpenRouter API.
        api_key (str, optional): API key for authentication.
    """
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        response = requests.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        models = response.json().get("data", [])
        if exclude_routers:
            models = [model for model in models if "router" not in model.get("id", "").lower()]

        free_models = []
        for model in models:
            pricing = model.get("pricing", {})
            if float(pricing.get("prompt", "0")) == 0 and float(pricing.get("completion", "0")) == 0:
                free_models.append(model)

        return free_models

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def filter_models(models, name=None, min_context_length=None, provider=None, required_parameters=None):
    """
    Filters a list of models based on specified criteria.

    Args:
        models (list): The list of models to filter.
        name (str, optional): The name to filter by (case-insensitive).
        min_context_length (int, optional): The minimum context length to filter by.
        provider (str, optional): The provider to filter by (case-insensitive).
        required_parameters (list, optional): List of parameter names that must be supported by the model.

    Returns:
        list: The filtered list of models.
    """
    filtered_models = models

    if name:
        filtered_models = [model for model in filtered_models if name.lower() in model.get("name", "").lower()]
    
    if min_context_length:
        filtered_models = [model for model in filtered_models if model.get("context_length", 0) >= min_context_length]

    if provider:
        filtered_models = [model for model in filtered_models if provider.lower() in model.get("id", "").lower().split("/")[0]]
    
    if required_parameters:
        filtered_models = [
            model for model in filtered_models 
            if all(param in model.get('supported_parameters', []) 
                  for param in required_parameters)
        ]

    return filtered_models

def sort_models(models, sort_by='name', reverse=False):
    """
    Sorts a list of models by a specified field.

    Args:
        models (list): The list of models to sort.
        sort_by (str): The field to sort by (e.g., 'name', 'context_length').
        reverse (bool): Whether to sort in descending order.

    Returns:
        list: The sorted list of models.
    """
    return sorted(models, key=lambda x: x.get(sort_by, 0), reverse=reverse)

def main():
    parser = argparse.ArgumentParser(description="Fetch and save free models from OpenRouter.")
    parser.add_argument("-o", "--output", help="Save the output to a JSON file.", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit the number of models returned.")
    parser.add_argument("--name", type=str, help="Filter models by name.")
    parser.add_argument("--min-context-length", type=int, help="Filter by minimum context length.")
    parser.add_argument("--provider", type=str, help="Filter by provider.")
    parser.add_argument("--require-params", type=str, help="Comma-separated list of required parameters (e.g., 'tool_choice,tools')")
    parser.add_argument("--sort-by", type=str, default="name", help="Sort models by a specific field (e.g., name, context_length).")
    parser.add_argument("--reverse", action="store_true", help="Reverse the sort order.")
    parser.add_argument("--base-url", type=str, default=OPENROUTER_DEFAULT_BASE_URL, help="Base URL for OpenRouter API.")
    parser.add_argument("--api-key", type=str, help="API key for OpenRouter authentication.")
    args = parser.parse_args()

    models = get_free_models(base_url=args.base_url, api_key=args.api_key)

    if models:
        # Parse required parameters
        required_params = None
        if args.require_params:
            required_params = [p.strip() for p in args.require_params.split(',') if p.strip()]
            print(f"Requiring models to support parameters: {', '.join(required_params)}")
            
        models = filter_models(
            models, 
            name=args.name, 
            min_context_length=args.min_context_length, 
            provider=args.provider,
            required_parameters=required_params
        )
        models = sort_models(models, sort_by=args.sort_by, reverse=args.reverse)

        if args.limit:
            models = models[:args.limit]

        if args.output:
            with open("free_models.json", "w") as f:
                json.dump(models, f, indent=2)
            print(f"Successfully saved {len(models)} models to free_models.json")
        else:
            print(json.dumps(models, indent=2))

if __name__ == "__main__":
    main()