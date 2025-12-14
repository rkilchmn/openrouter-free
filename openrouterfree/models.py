"""
Shared model utilities and statistics tracking for OpenRouter free scanner.
"""
import time
from typing import List, Dict, Any, Optional
# Moved imports inside functions to avoid circular imports

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class ModelStats:
    """Track error statistics for each model."""

    def __init__(self, error_threshold=3):
        self.stats = {}  # model_id -> {'errors': count, 'successes': count, 'last_error': timestamp}
        self.error_threshold = error_threshold

    def record_error(self, model_id):
        """Record an error for a model."""
        if model_id not in self.stats:
            self.stats[model_id] = {'errors': 0, 'successes': 0, 'last_error': None}

        self.stats[model_id]['errors'] += 1
        self.stats[model_id]['last_error'] = time.time()

    def record_success(self, model_id):
        """Record a successful request for a model."""
        if model_id not in self.stats:
            self.stats[model_id] = {'errors': 0, 'successes': 0, 'last_error': None}

        self.stats[model_id]['successes'] += 1

    def is_model_available(self, model_id):
        """Check if a model is available based on error threshold."""
        if model_id not in self.stats:
            return True

        stats = self.stats[model_id]

        # If last error was more than 5 minutes ago, reset error count
        if stats['last_error'] and (time.time() - stats['last_error']) > 300:
            stats['errors'] = 0
            return True

        return stats['errors'] < self.error_threshold

    def get_best_model(self, models):
        """Get the best model based on success rate and availability."""
        available_models = [m for m in models if self.is_model_available(m['id'])]

        if not available_models:
            # Reset all stats if no models are available
            self.stats = {}
            return models[0] if models else None

        # Sort by success rate (successes / (successes + errors))
        def success_rate(model):
            model_id = model['id']
            if model_id not in self.stats:
                return 1.0  # New models get priority

            stats = self.stats[model_id]
            total = stats['successes'] + stats['errors']
            if total == 0:
                return 1.0

            return stats['successes'] / total

        return max(available_models, key=success_rate)


def get_filtered_models(limit: Optional[int] = None,
                        name: Optional[str] = None,
                        min_context_length: Optional[int] = None,
                        provider: Optional[str] = None,
                        sort_by: str = 'context_length',
                        reverse: bool = True,
                        required_parameters: Optional[List[str]] = None,
                        base_url: str = OPENROUTER_DEFAULT_BASE_URL,
                        api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get and filter free models from OpenRouter.

    Args:
        limit: Maximum number of models to return
        name: Filter by model name
        min_context_length: Minimum context length required
        provider: Filter by provider
        sort_by: Field to sort by
        reverse: Reverse sort order
        required_parameters: Required parameters list
        base_url: Base URL for OpenRouter API
        api_key: API key for authentication

    Returns:
        List of filtered and sorted models
    """
    from .scanner import get_free_models, filter_models, sort_models
    
    models = get_free_models(base_url=base_url, api_key=api_key)

    if not models:
        raise ValueError("Could not fetch models from OpenRouter")

    # Apply filters
    models = filter_models(
        models,
        name=name,
        min_context_length=min_context_length,
        provider=provider,
        required_parameters=required_parameters
    )
    models = sort_models(models, sort_by=sort_by, reverse=reverse)

    if limit:
        models = models[:limit]

    if not models:
        raise ValueError("No models match the specified criteria")

    return models


def get_best_free_model(limit: Optional[int] = None,
                        name: Optional[str] = None,
                        min_context_length: Optional[int] = None,
                        provider: Optional[str] = None,
                        sort_by: str = 'context_length',
                        reverse: bool = True,
                        required_parameters: Optional[List[str]] = None,
                        base_url: str = OPENROUTER_DEFAULT_BASE_URL,
                        api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the single best free model based on filters.

    Returns the first model from the filtered and sorted list.
    """
    models = get_filtered_models(
        limit=limit,
        name=name,
        min_context_length=min_context_length,
        provider=provider,
        sort_by=sort_by,
        reverse=reverse,
        required_parameters=required_parameters,
        base_url=base_url,
        api_key=api_key
    )

    if not models:
        raise ValueError("No free models available")

    return models[0]