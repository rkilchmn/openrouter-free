from openai import OpenAI
from langchain_openai import ChatOpenAI
from .models import get_best_free_model, ModelStats
from typing import Optional, List
import time
import random

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterFreeOpenAIClient(ChatOpenAI):
    """
    LangChain-compatible OpenAI client that routes to free OpenRouter models.
    Inherits from ChatOpenAI for full method compatibility, with automatic best model selection and retry logic.
    """

    def __init__(self,
                 api_key: str,
                 base_url: str = OPENROUTER_DEFAULT_BASE_URL,
                 limit: Optional[int] = None,
                 name: Optional[str] = None,
                 min_context_length: Optional[int] = None,
                 provider: Optional[str] = None,
                 sort_by: str = 'context_length',
                 reverse: bool = True,
                 required_parameters: Optional[List[str]] = None,
                 max_retries: int = 3,
                 base_retry_delay: float = 1.0,
                 **kwargs):
        """
        Initialize the client.

        Args:
            api_key: OpenRouter API key
            base_url: Base URL for OpenRouter API (default: https://openrouter.ai/api/v1)
            limit: Limit number of models to use
            name: Filter models by name
            min_context_length: Minimum context length filter
            provider: Filter by provider
            sort_by: Field to sort models by
            reverse: Reverse sort order
            required_parameters: List of required parameters
            max_retries: Maximum number of retry attempts for failed requests
            base_retry_delay: Base delay in seconds for exponential backoff
        """
        # Store configuration for model selection
        config = {
            'limit': limit,
            'name': name,
            'min_context_length': min_context_length,
            'provider': provider,
            'sort_by': sort_by,
            'reverse': reverse,
            'required_parameters': required_parameters,
            'base_url': base_url,
            'api_key': api_key
        }

        # Initialize model statistics tracking
        error_threshold = max_retries
        model_stats = ModelStats(error_threshold=error_threshold)
        max_retries_val = max_retries
        base_retry_delay_val = base_retry_delay

        # Get the best free model
        best_model = self._get_best_free_model_with_stats_static(config, model_stats)

        if not best_model:
            raise ValueError("No free models available")

        # Use the best model
        print(f"Selected best free model: {best_model['id']}")

        # Call parent __init__ with selected model
        super().__init__(
            model=best_model['id'],
            api_key=api_key,
            base_url=base_url,
            max_retries=0,  # Disable parent's retries, we handle our own
            default_headers={
                'HTTP-Referer': 'https://github.com/tcsenpai/openrouter-free',
                'X-Title': 'OpenRouter Free Client'
            },
            **kwargs
        )

        # Now set instance attributes after super().__init__ (bypass Pydantic)
        object.__setattr__(self, 'config', config)
        object.__setattr__(self, 'error_threshold', error_threshold)
        object.__setattr__(self, 'model_stats', model_stats)
        object.__setattr__(self, 'max_retries', max_retries_val)
        object.__setattr__(self, 'base_retry_delay', base_retry_delay_val)
        object.__setattr__(self, 'best_model', best_model)

    @staticmethod
    def _get_best_free_model_with_stats_static(config, model_stats):
        """Get the best available free model using statistics tracking."""
        try:
            # Get filtered models using the stored configuration
            from .models import get_filtered_models
            models = get_filtered_models(**config)

            if not models:
                return None

            # Use ModelStats to get the best model
            return model_stats.get_best_model(models)
        except ValueError:
            return None

    def _get_best_free_model(self, limit=None, name=None, min_context_length=None,
                            provider=None, sort_by='context_length', reverse=True,
                            required_parameters=None, base_url=OPENROUTER_DEFAULT_BASE_URL,
                            api_key=None):
        """Get the best available free model."""
        try:
            return get_best_free_model(
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
        except ValueError:
            return None
            
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Override to add retry logic with model switching."""
        last_exception = None

        while True:
            if not isinstance(self.best_model, dict) or 'id' not in self.best_model:
                raise ValueError("No valid best model available")

            model_id = self.best_model['id']
            print(f"Trying model: {model_id}")

            # Check if this model has already exceeded error threshold
            if not self.model_stats.is_model_available(model_id):
                print(f"Model {model_id} has exceeded error threshold ({self.error_threshold} errors), trying alternative...")
                new_model = self._get_best_free_model_with_stats_static(self.config, self.model_stats)

                if not new_model or new_model['id'] == model_id:
                    print("No better alternative model available")
                    break

                object.__setattr__(self, 'best_model', new_model)
                self.model = new_model['id']  # Update model in parent
                print(f"Switched to alternative model: {new_model['id']}")
                continue

            # Try current model up to max_retries times
            for attempt in range(self.max_retries + 1):
                try:
                    # Call parent's _generate
                    result = super()._generate(messages, stop, run_manager, **kwargs)

                    # Record success for the actual model used (may differ due to routing)
                    actual_model = result.generations[0].message.response_metadata.get('model', model_id)
                    self.model_stats.record_success(actual_model)
                    print(f"Success with model: {actual_model}")
                    return result

                except Exception as e:
                    last_exception = e

                    # Record error for current model
                    self.model_stats.record_error(model_id)

                    # Check if this model has now exceeded error threshold
                    if not self.model_stats.is_model_available(model_id):
                        print(f"Model {model_id} has now exceeded error threshold ({self.error_threshold} errors)")
                        break

                    # Check if this was the last retry for this attempt
                    if attempt >= self.max_retries:
                        print(f"All {self.max_retries} retry attempts exhausted for this API call")
                        break

                    # Don't wait before the last attempt
                    if attempt >= self.max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {str(e)[:100]}...")
                        continue  # Skip delay and retry immediately

                    # Calculate exponential backoff delay with jitter
                    delay = self.base_retry_delay * (2 ** attempt)
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter

                    print(f"Attempt {attempt + 1} failed: {str(e)[:100]}...")
                    print(f"Retrying in {total_delay:.1f} seconds...")
                    time.sleep(total_delay)

            # Check again if model exceeded error threshold after this round of retries
            if not self.model_stats.is_model_available(model_id):
                print(f"Model {model_id} exceeded error threshold after retries, trying alternative...")
                new_model = self._get_best_free_model_with_stats_static(self.config, self.model_stats)

                if not new_model or new_model['id'] == model_id:
                    print("No better alternative model available")
                    break

                object.__setattr__(self, 'best_model', new_model)
                self.model = new_model['id']
                print(f"Switched to alternative model: {new_model['id']}")
                # Continue the outer while loop to retry with new model
            else:
                # Model still available but all retries exhausted, try alternative anyway
                print(f"All retries exhausted for {model_id}, trying alternative model...")
                new_model = self._get_best_free_model_with_stats_static(self.config, self.model_stats)

                if not new_model or new_model['id'] == model_id:
                    print("No better alternative model available")
                    break

                object.__setattr__(self, 'best_model', new_model)
                self.model = new_model['id']
                print(f"Switched to alternative model: {new_model['id']}")
                # Continue the outer while loop to retry with new model

        # All models exhausted
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed")



