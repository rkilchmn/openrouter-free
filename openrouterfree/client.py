from openai import OpenAI
from .models import get_best_free_model, ModelStats
from typing import Optional, List
import time
import random

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterFreeOpenAIClient:
    """
    Simplified OpenAI-compatible client that routes to free OpenRouter models.
    Uses OpenAI client with OpenRouter URL and automatic best model selection.
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
                 base_retry_delay: float = 1.0):
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
        # Initialize model statistics tracking
        self.error_threshold = max_retries
        self.model_stats = ModelStats(error_threshold=self.error_threshold)
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        
        
        # Store configuration for retries
        self.config = {
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
        
        # Get the best free model
        self.best_model = self._get_best_free_model_with_stats()

        if not self.best_model:
            raise ValueError("No free models available")

        # Use the best model
        print(f"Selected best free model: {self.best_model['id']}")

        # Create OpenAI client configured for OpenRouter
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                'HTTP-Referer': 'https://github.com/tcsenpai/openrouter-free-scanner',
                'X-Title': 'OpenRouter Free Client'
            }
        )

        # Set up chat interface for compatibility
        self.chat = self.Chat(self)
        self.completions = self.chat

    def _get_best_free_model_with_stats(self):
        """Get the best available free model using statistics tracking."""
        try:
            # Get filtered models using the stored configuration
            from .models import get_filtered_models
            models = get_filtered_models(**self.config)
            
            if not models:
                return None
                
            # Use ModelStats to get the best model
            return self.model_stats.get_best_model(models)
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
            
    def _retry_with_exponential_backoff(self, func, *args, **kwargs):
        """Execute function with exponential retry and model switching based on error threshold."""
        last_exception = None
        
        # Keep trying until we get a successful result or exhaust all models
        while True:
            if not isinstance(self.best_model, dict) or 'id' not in self.best_model:
                raise ValueError("No valid best model available")
                
            model_id = self.best_model['id']
            print(f"Trying model: {model_id}")
            
            # Check if this model has already exceeded error threshold
            if not self.model_stats.is_model_available(model_id):
                print(f"Model {model_id} has exceeded error threshold ({self.error_threshold} errors), trying alternative...")
                new_model = self._get_best_free_model_with_stats()
                
                if not new_model or new_model['id'] == model_id:
                    print("No better alternative model available")
                    break
                    
                self.best_model = new_model
                print(f"Switched to alternative model: {new_model['id']}")
                continue
            
            # Try current model up to max_retries times
            for attempt in range(self.max_retries + 1):
                try:
                    # Try to execute the function
                    result = func(*args, **kwargs)
                    
                    # Record success for current best model
                    self.model_stats.record_success(model_id)
                    print(f"Success with model: {model_id}")
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
                new_model = self._get_best_free_model_with_stats()
                
                if not new_model or new_model['id'] == model_id:
                    print("No better alternative model available")
                    break
                    
                self.best_model = new_model
                print(f"Switched to alternative model: {new_model['id']}")
                # Continue the outer while loop to retry with new model
            else:
                # Model still available but all retries exhausted, try alternative anyway
                print(f"All retries exhausted for {model_id}, trying alternative model...")
                new_model = self._get_best_free_model_with_stats()
                
                if not new_model or new_model['id'] == model_id:
                    print("No better alternative model available")
                    break
                    
                self.best_model = new_model
                print(f"Switched to alternative model: {new_model['id']}")
                # Continue the outer while loop to retry with new model
        
        # All models exhausted
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed")

    class Chat:
        """Chat completions interface."""

        def __init__(self, parent_client):
            self.parent = parent_client
            self.completions = self

        def create(self, model="auto", messages=None, **kwargs):
            """
            Create chat completion using the pre-selected best model.
            Ignores the model parameter and uses the best free model.
            """
            return self.parent.chat_completions_create(model, messages, **kwargs)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Create chat completion using OpenAI client with OpenRouter.
        Always uses the pre-selected best free model with exponential retry.

        Args:
            model: Ignored - always uses the best free model
            messages: List of message dictionaries
            **kwargs: Additional OpenAI API parameters

        Returns:
            OpenAI ChatCompletion response
        """
        # Ensure we have a valid best model
        if not self.best_model or not isinstance(self.best_model, dict) or 'id' not in self.best_model:
            raise ValueError("No valid best model available")
            
        model_id = self.best_model['id']
        
        # Use exponential retry for the API call
        def make_api_call():
            return self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                **kwargs
            )
            
        return self._retry_with_exponential_backoff(make_api_call)


