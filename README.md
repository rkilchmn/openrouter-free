# OpenRouter Free

A powerful tool to discover and interact with free AI models from OpenRouter.ai. Includes a command-line interface, Python library, and an OpenAI-compatible proxy server for seamless integration.

## Features

- **Model Discovery**: Find and filter free AI models by name, provider, and capabilities
- **Proxy Server**: Run an OpenAI-compatible API server that automatically routes requests to available free models
- **Automatic Failover**: Built-in retry logic with model failover for reliable operation
- **Parameter Filtering**: Select models based on supported parameters (e.g., function calling, tool use)

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/openrouter-free.git
cd openrouter-free
pip install .

# Or install directly with pip
pip install openrouter-free
```

## Basic Usage

### Command Line

List available free models:

```bash
openrouter-free
```

Save models to a JSON file:

```bash
openrouter-free -o
```

### Python Library

```python
from openrouterfree import get_free_models, filter_models, sort_models

# Get all free models
models = get_free_models()

# Filter models by capabilities
tools_models = filter_models(
    models, 
    required_parameters=['tool_choice', 'tools'],
    min_context_length=8000
)

# Sort by context length
sorted_models = sort_models(models, sort_by='context_length', reverse=True)
```

### OpenRouterFreeOpenAIClient

For seamless integration with existing OpenAI-based code, use the `OpenRouterFreeOpenAIClient` which automatically selects and uses the best available free model:

```python
from openrouterfree import OpenRouterFreeOpenAIClient

# Create client - automatically selects best free model
client = OpenRouterFreeOpenAIClient(
    api_key='your-openrouter-api-key'
)

# Create client with additional OpenAI parameters
client = OpenRouterFreeOpenAIClient(
    api_key='your-openrouter-api-key',
    timeout=30.0  # Pass through OpenAI client parameters
)

# Use like standard OpenAI client
response = client.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
```

## Proxy Server

The proxy server provides an OpenAI-compatible API endpoint that automatically routes requests to available free models.

### Starting the Proxy

```bash
openrouter-free-proxy --port 8080
```

### Advanced Proxy Options

```bash
openrouter-free-proxy \
  --port 8080 \
  --limit 5 \
  --require-params "tool_choice,tools" \
  --min-context-length 4000 \
  --sort-by "context_length" \
  --reverse
```

### Using with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-openrouter-api-key"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Will be overridden by proxy
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Proxy API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)
- `GET /health` - Health check endpoint

## Advanced Features

### Model Filtering

Filter models by supported parameters:

```python
# Only include models that support function calling
function_models = filter_models(
    models,
    required_parameters=['functions', 'function_call']
)
```

### Automatic Retry & Failover

The proxy automatically handles rate limits and errors by:
1. Retrying failed requests with the same model
2. Failing over to the next best available model
3. Tracking model health and success rates

### Rate Limiting

The proxy respects rate limits and will automatically:
- Back off when rate limited
- Rotate to other available models
- Return appropriate rate limit headers to clients

## Command Line Reference

### Scanner Options

```
usage: openrouter-free [-h] [-o] [--limit LIMIT] [--name NAME]
                              [--min-context-length MIN_CONTEXT_LENGTH]
                              [--provider PROVIDER] [--sort-by SORT_BY]
                              [--reverse] [--require-params REQUIRE_PARAMS]

options:
  -h, --help            show this help message and exit
  -o, --output          Save output to free_models.json
  --limit LIMIT         Limit number of models returned
  --name NAME           Filter models by name
  --min-context-length MIN_CONTEXT_LENGTH
                        Filter by minimum context length
  --provider PROVIDER   Filter by provider
  --sort-by SORT_BY     Field to sort by (default: name)
  --reverse             Reverse sort order
  --require-params REQUIRE_PARAMS
                        Comma-separated list of required parameters
```

### Proxy Options

```
usage: openrouter-free-proxy [-h] [--port PORT] [--limit LIMIT] [--name NAME]
               [--min-context-length MIN_CONTEXT_LENGTH] [--provider PROVIDER]
               [--sort-by SORT_BY] [--reverse] [--error-threshold ERROR_THRESHOLD]
               [--require-params REQUIRE_PARAMS]

options:
  -h, --help            show this help message and exit
  --port PORT           Port to run the server on (default: 8080)
  --limit LIMIT         Limit number of models to use
  --name NAME           Filter models by name
  --min-context-length MIN_CONTEXT_LENGTH
                        Filter by minimum context length
  --provider PROVIDER   Filter by provider
  --sort-by SORT_BY     Sort models by field (default: context_length)
  --reverse             Reverse sort order (default: True)
  --error-threshold ERROR_THRESHOLD
                        Number of errors before switching models (default: 3)
  --require-params REQUIRE_PARAMS
                        Comma-separated list of required parameters
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.