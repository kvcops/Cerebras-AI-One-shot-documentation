# Cerebras Inference API - Technical Reference

## Document Metadata

- **Document Type**: LLM Knowledge Base Reference
- **API Base URL**: `https://api.cerebras.ai/v1`
- **Authentication**: Bearer token via `Authorization: Bearer <CEREBRAS_API_KEY>`
- **SDK Packages**: 
  - Python: `cerebras_cloud_sdk`
  - Node.js: `@cerebras/cerebras_cloud_sdk`

---

# 1. Overview

## 1.1 What Cerebras Is

Cerebras provides high-performance AI inference infrastructure. The Cerebras Inference API serves large language models (LLMs) with high throughput and low latency. The API is designed to be OpenAI-compatible, allowing existing applications to migrate with minimal code changes.

## 1.2 Core Capabilities

- **Chat Completions**: Multi-turn conversation generation
- **Streaming**: Token-by-token incremental responses
- **Tool Calling**: Function calling with schema enforcement
- **Structured Outputs**: JSON schema-constrained generation
- **Reasoning**: Extended thinking with configurable visibility
- **Predicted Outputs**: Accelerated generation for known content
- **Prompt Caching**: Automatic prefix caching for reduced latency

---

# 2. Models

## 2.1 Production Models

Production models are fully supported and intended for production environments.

| Model Name | Model ID | Parameters | Speed | Deprecation |
|------------|----------|------------|-------|-------------|
| Llama 3.1 8B | `llama3.1-8b` | 8B | ~2200 tokens/s | Active |
| Llama 3.3 70B | `llama-3.3-70b` | 70B | ~2100 tokens/s | February 16, 2026 |
| OpenAI GPT OSS | `gpt-oss-120b` | 120B | ~3000 tokens/s | Active |
| Qwen 3 32B | `qwen-3-32b` | 32B | ~2600 tokens/s | February 16, 2026 |

## 2.2 Preview Models

Preview models are for evaluation only. They may be discontinued with short notice.

| Model Name | Model ID | Parameters | Speed |
|------------|----------|------------|-------|
| Qwen 3 235B Instruct | `qwen-3-235b-a22b-instruct-2507` | 235B | ~1400 tokens/s |
| Z.ai GLM 4.7 | `zai-glm-4.7` | 355B (32B active) | ~1000 tokens/s |

## 2.3 Model Specifications

### 2.3.1 gpt-oss-120b (OpenAI GPT OSS)

**Capabilities**: Reasoning, Streaming, Structured Outputs, Tool Calling, Prompt Caching

**Context Length**:
- Free Tier: 65k tokens
- Paid Tiers: 131k tokens

**Max Output**:
- Free Tier: 32k tokens
- Paid Tiers: 40k tokens

**Pricing**:
- Input: $0.35 / million tokens
- Output: $0.75 / million tokens

**Rate Limits**:
| Tier | Requests/min | Input Tokens/min | Daily Tokens |
|------|--------------|------------------|--------------|
| Free | 30 | 60k | 1M |
| Developer | 1K | 1M | Unlimited |

**Model Notes**:
- Use `reasoning_effort` parameter to control reasoning. Default: `medium`. Options: `low`, `medium`, `high`.
- When `min_tokens` is set, the model may generate EOS tokens causing parser failures.
- The model may call tools not specified in the request. Monitor for non-approved tools and reprompt with "you're hallucinating a tool call" to correct.
- The `system` role maps to developer-level instructions in the prompt hierarchy.
- Precision: FP16/FP8 (weights only, selective quantization with dequantization on-the-fly)

### 2.3.2 zai-glm-4.7 (Z.ai GLM 4.7)

**Architecture**: Mixture-of-Experts (MoE) Transformer. 358B total parameters, ~32B active per forward pass.

**Capabilities**: Reasoning, Streaming, Structured Outputs, Tool Calling, Parallel Tool Calling, Prompt Caching

**Context Length**:
- Free Tier: 64k tokens
- Paid Tiers: 131k tokens

**Max Output**:
- Free Tier: 40k tokens
- Paid Tiers: 40k tokens

**Pricing**:
- Input: $2.25 / million tokens
- Output: $2.75 / million tokens

**Rate Limits**:
| Tier | Requests/min | Input Tokens/min | Daily Tokens |
|------|--------------|------------------|--------------|
| Free | 10 | 60k | 1M |
| Developer | 500 | 500k | Unlimited |

**Model Notes**:
- Reasoning is enabled by default. Use `disable_reasoning: true` to disable.
- Structured outputs and tool calling with `strict: true` (constrained decoding) is supported.
- Use `clear_thinking` parameter to control whether thinking content from previous turns is included. Default: `true` (exclude previous thinking). Set to `false` for agentic workflows.
- Precision: FP16/FP8 (weights only)
- License: MIT-style permissive license

### 2.3.3 llama3.1-8b

**Capabilities**: Streaming, Tool Calling, Predicted Outputs, Prompt Caching

**Context Length**:
- Free Tier: 8k tokens
- Paid Tiers: 128k tokens

**Precision**: FP16

### 2.3.4 llama-3.3-70b

**Capabilities**: Streaming, Tool Calling, Prompt Caching

**Context Length**:
- Free Tier: 8k tokens
- Paid Tiers: 128k tokens

**Precision**: FP16

### 2.3.5 qwen-3-32b

**Capabilities**: Reasoning, Streaming, Tool Calling, Prompt Caching

**Context Length**:
- Free Tier: 8k tokens
- Paid Tiers: 131k tokens

**Precision**: FP16

**Model Notes**:
- Default reasoning format: `raw` (uses `<think>...</think>` tags)
- When using JSON object/schema response format: reasoning format defaults to `hidden`

## 2.4 Model Compression Details

All models served through public endpoints are unpruned. Selective weight-only quantization is used during storage for some models (FP16/FP8 with sensitive layers stored at full precision). Dequantization occurs on-the-fly. Activations and KV cache remain in higher precision and unquantized.

| Model ID | Precision |
|----------|-----------|
| `llama3.1-8b` | FP16 |
| `llama-3.3-70b` | FP16 |
| `gpt-oss-120b` | FP16/FP8 (weights only) |
| `qwen-3-32b` | FP16 |
| `qwen-3-235b-a22b-instruct-2507` | FP16/FP8 (weights only) |
| `zai-glm-4.7` | FP16/FP8 (weights only) |

---

# 3. Installation & Setup

## 3.1 Prerequisites

- Cerebras account
- Cerebras API key
- Python 3.7+ or TypeScript 4.5+

## 3.2 API Key Configuration

Set the API key as an environment variable:

```bash
export CEREBRAS_API_KEY="your-api-key-here"
```

## 3.3 SDK Installation

### Python

```bash
pip install --upgrade cerebras_cloud_sdk
```

### Node.js

```bash
npm install @cerebras/cerebras_cloud_sdk@latest
```

## 3.4 Client Initialization

### Python

```python
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)
```

### Node.js

```javascript
import Cerebras from '@cerebras/cerebras_cloud_sdk';

const client = new Cerebras({
    apiKey: process.env['CEREBRAS_API_KEY'],
});
```

---

# 4. API Reference

## 4.1 Chat Completions Endpoint

**Endpoint**: `POST /v1/chat/completions`

### 4.1.1 Request Structure

```json
{
    "model": "string (required)",
    "messages": [
        {
            "role": "system" | "user" | "assistant" | "tool",
            "content": "string",
            "tool_call_id": "string (for tool role)"
        }
    ],
    "stream": false,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_completion_tokens": -1,
    "seed": 0,
    "tools": [],
    "tool_choice": "auto" | "none" | {"type": "function", "function": {"name": "string"}},
    "parallel_tool_calls": true,
    "response_format": {"type": "text" | "json_object" | "json_schema"},
    "prediction": {"type": "content", "content": "string"},
    "reasoning_effort": "low" | "medium" | "high",
    "reasoning_format": "parsed" | "raw" | "hidden" | "none",
    "disable_reasoning": false,
    "clear_thinking": true,
    "logprobs": false
}
```

### 4.1.2 Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model ID to use |
| `messages` | array | required | Conversation messages array |
| `stream` | boolean | false | Enable streaming responses |
| `temperature` | float | 1.0 | Sampling temperature (0.0 to 2.0) |
| `top_p` | float | 1.0 | Nucleus sampling probability |
| `max_completion_tokens` | integer | -1 | Maximum output tokens. -1 uses model maximum |
| `seed` | integer | 0 | Random seed for reproducibility |

### 4.1.3 Tool Calling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tools` | array | [] | Tool definitions with function schemas |
| `tool_choice` | string/object | "auto" | Tool selection behavior |
| `parallel_tool_calls` | boolean | true | Allow multiple simultaneous tool calls |

### 4.1.4 Response Format Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_format` | object | {"type": "text"} | Response format specification |
| `response_format.type` | string | "text" | Format type: text, json_object, json_schema |
| `response_format.json_schema` | object | - | JSON schema for json_schema type |
| `response_format.json_schema.strict` | boolean | false | Enable constrained decoding |

### 4.1.5 Reasoning Parameters

| Parameter | Type | Default | Models | Description |
|-----------|------|---------|--------|-------------|
| `reasoning_effort` | string | "medium" | gpt-oss-120b | Reasoning intensity: low, medium, high |
| `reasoning_format` | string | "none" | All reasoning models | Reasoning output format |
| `disable_reasoning` | boolean | false | zai-glm-4.7 | Disable reasoning entirely |
| `clear_thinking` | boolean | true | zai-glm-4.7 | Exclude previous thinking from context |

### 4.1.6 Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction` | object | - | Predicted output for accelerated generation |
| `prediction.type` | string | "content" | Always "content" |
| `prediction.content` | string | - | Expected output content |

### 4.1.7 Response Structure (Non-Streaming)

```json
{
    "id": "string",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "string",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "string",
                "reasoning": "string (if reasoning_format: parsed)",
                "tool_calls": [
                    {
                        "id": "string",
                        "type": "function",
                        "function": {
                            "name": "string",
                            "arguments": "string (JSON)"
                        }
                    }
                ]
            },
            "logprobs": {},
            "reasoning_logprobs": {},
            "finish_reason": "stop" | "length" | "tool_calls"
        }
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0
        }
    }
}
```

### 4.1.8 Response Structure (Streaming)

Each streamed chunk:

```json
{
    "id": "string",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "string",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "string",
                "reasoning": "string",
                "tool_calls": []
            },
            "finish_reason": null | "stop" | "length" | "tool_calls"
        }
    ]
}
```

---

# 5. Streaming

## 5.1 Enabling Streaming

Set `stream: true` in the request. The response is an iterable of chunks.

### Python

```python
stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Why is fast inference important?"}],
    model="llama-3.3-70b",
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Node.js

```javascript
const stream = await client.chat.completions.create({
    messages: [{ role: 'user', content: 'Why is fast inference important?' }],
    model: 'llama-3.3-70b',
    stream: true,
});

for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

## 5.2 Streaming Constraints

- Streaming is NOT supported when using reasoning models with JSON mode or tool calling
- Streaming IS supported for `gpt-oss-120b`, `zai-glm-4.7`, and non-reasoning models with these features

---

# 6. Reasoning

## 6.1 Reasoning Models

Reasoning is available for:
- `gpt-oss-120b`
- `qwen-3-32b`
- `zai-glm-4.7`

## 6.2 Reasoning Format

The `reasoning_format` parameter controls how reasoning text appears in responses.

| Format | Description |
|--------|-------------|
| `parsed` | Reasoning returned in separate `reasoning` field; logprobs in `reasoning_logprobs` |
| `raw` | Reasoning prepended to content. GLM/Qwen use `<think>...</think>` tags. GPT-OSS concatenates without separators |
| `hidden` | Reasoning text and logprobs dropped completely. Tokens still counted in usage |
| `none` | Uses model's default behavior |

### 6.2.1 Default Behavior by Model

| Model | Default Reasoning Format |
|-------|-------------------------|
| Qwen3 | `raw` (`hidden` for JSON object/schema) |
| GLM | `parsed` |
| GPT-OSS | `parsed` |

### 6.2.2 Parsed Format Example

```python
response = client.chat.completions.create(
    model="zai-glm-4.7",
    messages=[{"role": "user", "content": "Can you help me with this?"}],
    reasoning_format="parsed"
)

# Response includes message.reasoning and message.content separately
# If streaming: delta.reasoning contains reasoning tokens
```

### 6.2.3 Raw Format Example

```python
response = client.chat.completions.create(
    model="zai-glm-4.7",
    messages=[{"role": "user", "content": "Can you help me with this?"}],
    reasoning_format="raw"
)

# Response content includes: "<think>Let me think...</think>I can help you with that!"
```

**Constraint**: `raw` format is NOT compatible with `json_object` or `json_schema` response formats.

## 6.3 Model-Specific Reasoning Parameters

### 6.3.1 GPT-OSS: reasoning_effort

Controls reasoning intensity.

| Value | Behavior |
|-------|----------|
| `low` | Minimal reasoning, faster responses |
| `medium` | Moderate reasoning (default) |
| `high` | Extensive reasoning, thorough analysis |

```python
response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "Explain quantum entanglement."}],
    reasoning_effort="medium"
)
```

### 6.3.2 GLM: disable_reasoning

Toggle reasoning on or off.

```python
response = client.chat.completions.create(
    model="zai-glm-4.7",
    messages=[{"role": "user", "content": "Explain how photosynthesis works."}],
    disable_reasoning=False  # True to disable
)
```

### 6.3.3 GLM: clear_thinking

Controls whether thinking content from previous turns is included in context.

| Value | Behavior |
|-------|----------|
| `true` (default) | Exclude previous thinking from context |
| `false` | Include previous thinking. Recommended for agentic workflows |

## 6.4 Reasoning Context Retention

Reasoning tokens are NOT automatically retained across requests. To maintain awareness of prior reasoning in multi-turn conversations, include the reasoning text in the assistant message's `content` field.

### GPT-OSS Format

Prepend reasoning directly before the answer:

```python
messages=[
    {"role": "user", "content": "What is 25 * 4?"},
    {"role": "assistant", "content": "I need to multiply 25 by 4. 25 * 4 = 100. The answer is 100."},
    {"role": "user", "content": "Now divide that by 2."}
]
```

### GLM/Qwen Format

Wrap reasoning in `<think>` tags:

```python
messages=[
    {"role": "user", "content": "What is 25 * 4?"},
    {"role": "assistant", "content": "<think>I need to multiply 25 by 4. 25 * 4 = 100.</think>The answer is 100."},
    {"role": "user", "content": "Now divide that by 2."}
]
```

---

# 7. Tool Calling

## 7.1 Tool Definition Structure

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "function_name",
            "strict": True,  # Enable constrained decoding for arguments
            "description": "Description of what the function does",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param1"],
                "additionalProperties": False  # Required when strict: true
            }
        }
    }
]
```

## 7.2 Tool Calling Workflow

1. **Define tools**: Provide name, description, and parameter schema
2. **Send request**: Include tools array in API call
3. **Model decides**: Model returns `tool_calls` if tools are needed
4. **Execute tool**: Client executes the tool and retrieves result
5. **Return result**: Send tool result back with `role: "tool"` and `tool_call_id`
6. **Final response**: Model generates final response using tool results

## 7.3 Basic Tool Calling Example

```python
# Define tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "strict": True,
            "description": "Performs basic arithmetic operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    }
]

# Initial request
response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant with access to a calculator."},
        {"role": "user", "content": "What's the result of 15 multiplied by 7?"},
    ],
    tools=tools,
    parallel_tool_calls=False,
)

# Process tool calls
choice = response.choices[0].message
if choice.tool_calls:
    tool_call = choice.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    result = calculate(arguments["expression"])
    
    # Send result back
    messages.append({
        "role": "tool",
        "content": json.dumps(result),
        "tool_call_id": tool_call.id
    })
    
    # Get final response
    final_response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=messages,
    )
```

## 7.4 Multi-Turn Tool Calling

Loop until the model responds without `tool_calls`:

```python
while True:
    response = client.chat.completions.create(
        model="qwen-3-32b",
        messages=messages,
        tools=tools,
    )
    msg = response.choices[0].message
    
    if not msg.tool_calls:
        print("Assistant:", msg.content)
        break
    
    messages.append(msg.model_dump())
    
    for call in msg.tool_calls:
        result = execute_tool(call.function.name, json.loads(call.function.arguments))
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
```

## 7.5 Parallel Tool Calling

Enable with `parallel_tool_calls: true` (default). The model can request multiple tools simultaneously.

```python
response = client.chat.completions.create(
    model="zai-glm-4.7",
    messages=[{"role": "user", "content": "Is Toronto warmer than Montreal?"}],
    tools=[get_weather_tool],
    parallel_tool_calls=True,
)

# response.choices[0].message.tool_calls may contain multiple items
for tool_call in response.choices[0].message.tool_calls:
    # Process each tool call
    pass
```

## 7.6 Strict Mode for Tools

When `strict: true` is set in the function definition:
- Arguments are guaranteed to match schema exactly
- `additionalProperties: false` is required for all objects
- Constrained decoding ensures schema compliance

Without strict mode, tool calls may include:
- Wrong parameter types
- Missing required parameters
- Unexpected extra parameters
- Malformed argument JSON

---

# 8. Structured Outputs

## 8.1 JSON Schema Response Format

Enforce consistent JSON outputs using `response_format` with `json_schema`.

```python
movie_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "director": {"type": "string"},
        "year": {"type": "integer"},
    },
    "required": ["title", "director", "year"],
    "additionalProperties": False
}

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You generate movie recommendations."},
        {"role": "user", "content": "Suggest a sci-fi movie from the 1990s"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "movie_schema",
            "strict": True,
            "schema": movie_schema
        }
    }
)

movie_data = json.loads(response.choices[0].message.content)
```

## 8.2 Strict Mode

When `strict: true`:
- Guaranteed valid JSON
- Schema compliance enforced at token level
- Type safety guaranteed
- No retries needed for schema violations
- Constrained decoding is applied

### 8.2.1 Strict Mode Requirements

- `additionalProperties: false` is required for every object in the schema

### 8.2.2 Strict Mode Limitations

| Category | Limitation |
|----------|------------|
| Schema size | Maximum 5000 characters |
| Nesting depth | Maximum 10 levels |
| Object properties | Maximum 500 |
| Enum values | Maximum 500 across all properties |
| String enum length | Maximum 7500 characters total when >250 enum values |
| Recursive schemas | Not supported |
| `items: true` | Not supported for arrays |
| `$anchor` | Not supported |
| External references | Not supported (security) |

### 8.2.3 Supported Reference Patterns

- Internal definitions: `"$ref": "#/$defs/cast_member"`
- Use `$defs` instead of `definitions`

## 8.3 JSON Mode

Simpler alternative without schema enforcement:

```python
response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You generate movie recommendations."},
        {"role": "user", "content": "Suggest a sci-fi movie from the 1990s"}
    ],
    response_format={"type": "json_object"}
)
```

**Requirement**: You must explicitly instruct the model to generate JSON through a system or user message.

## 8.4 Structured Outputs vs JSON Mode Comparison

| Feature | Structured Outputs (strict) | Structured Outputs (non-strict) | JSON Mode |
|---------|---------------------------|-------------------------------|-----------|
| Outputs valid JSON | Yes | Yes (best-effort) | Yes |
| Adheres to schema | Yes (guaranteed) | Yes | No |
| Extra fields allowed | No | Yes | Flexible |
| Constrained Decoding | Yes | No | No |

**Constraint**: `tools` and `response_format` cannot be used in the same request.

---

# 9. Predicted Outputs

## 9.1 Overview

Predicted Outputs accelerate generation when parts of the output are known. The model reuses matching tokens and regenerates only those that differ.

## 9.2 Supported Models

- `gpt-oss-120b`
- `llama3.1-8b`
- `zai-glm-4.7`

## 9.3 Recommended Use Cases

- Code refactoring with minor changes
- Document editing (grammar fixes, tone adjustments)
- Template filling with placeholder updates

## 9.4 Usage

```python
code = """
html {
    margin: 0;
    padding: 0;
}
body {
    color: #00FF00;
}
"""

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Change the color to blue. Respond only with code."},
        {"role": "user", "content": code}
    ],
    prediction={"type": "content", "content": code},
)
```

## 9.5 Token-Reuse Metrics

Response includes usage metrics:

```json
{
    "usage": {
        "completion_tokens": 224,
        "prompt_tokens": 204,
        "total_tokens": 428,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 76,
            "rejected_prediction_tokens": 20
        }
    }
}
```

## 9.6 Best Practices

- Set `temperature=0` for higher token acceptance
- Keep predictions accurate to minimize rejected tokens
- Monitor acceptance rate; fall back to standard completion if rejection is high

## 9.7 Limitations

- `logprobs`: Not supported
- `n > 1`: Not supported
- `tools`: Not supported with predicted outputs
- Rejected prediction tokens are billed at completion-token rates
- Reasoning tokens may generate additional `rejected_prediction_tokens`

---

# 10. Prompt Caching

## 10.1 Mechanism

Prompt caching is automatic. No code changes required.

1. **Prefix Matching**: System analyzes prompt prefix (system prompts, tool definitions, few-shot examples)
2. **Block-Based Caching**: Prompts processed in blocks (100-600 tokens)
3. **Cache Hit**: Matching blocks skip processing, reducing latency
4. **Cache Miss**: Prompt processed normally; prefix stored for future matches
5. **Automatic Expiration**: TTL guaranteed 5 minutes, may persist up to 1 hour

## 10.2 Supported Models

- `zai-glm-4.7`
- `gpt-oss-120b`
- `qwen-3-235b-a22b-instruct-2507`
- `qwen-3-32b`
- `llama-3.3-70b`

## 10.3 Cache Match Requirements

The entire beginning of the prompt must match exactly. A single character difference in the first token causes a cache miss for that block and all subsequent blocks.

## 10.4 Prompt Structure for Optimal Caching

**Static content first**:
- System instructions
- Tool definitions and schemas
- Few-shot examples
- Large context documents

**Dynamic content last**:
- User-specific questions
- Session variables
- Timestamps

### Correct Structure

```json
[
    {"role": "system", "content": "You are a coding assistant... Current Time: 12:01 PM"},
    {"role": "user", "content": "Debug this code."}
]
```

### Incorrect Structure (Cache Miss)

```json
[
    {"role": "system", "content": "Current Time: 12:01 PM. You are a coding assistant..."},
    {"role": "user", "content": "Debug this code."}
]
```

## 10.5 Tracking Cache Usage

Check `usage.prompt_tokens_details.cached_tokens` in the response:

```json
"usage": {
    "prompt_tokens": 3000,
    "completion_tokens": 150,
    "total_tokens": 3150,
    "prompt_tokens_details": {
        "cached_tokens": 2800
    }
}
```

## 10.6 Caching Behavior

- Cached tokens count toward rate limits
- No additional fee for caching
- Caches are organization-scoped, never shared between organizations
- No manual cache management available
- Data is ephemeral, never persisted
- ZDR-compliant

---

# 11. Rate Limits

## 11.1 Rate Limit Metrics

- **RPM**: Requests per minute
- **RPH**: Requests per hour
- **RPD**: Requests per day
- **TPM**: Tokens per minute
- **TPH**: Tokens per hour
- **TPD**: Tokens per day

Rate limits apply at the organization level. Triggering occurs when any metric limit is reached.

## 11.2 Token Rate Limiting Calculation

When sending a request, the system estimates total token consumption:

1. Estimate input tokens in prompt
2. Add `max_completion_tokens` value (or maximum sequence length minus input tokens)

If estimated consumption exceeds quota, request is rate limited before processing.

**Best Practice**: Set `max_completion_tokens` appropriately to avoid overestimation.

## 11.3 Quota Replenishment

Uses token bucketing algorithm. Capacity replenishes continuously rather than resetting at fixed intervals.

```
Available quota = Rate limit - Usage in current time window
```

## 11.4 Free Tier Limits

| Model | TPM | TPH | TPD | RPM | RPH | RPD |
|-------|-----|-----|-----|-----|-----|-----|
| `gpt-oss-120b` | 60K | 1M | 1M | 30 | 900 | 14.4K |
| `llama3.1-8b` | 60K | 1M | 1M | 30 | 900 | 14.4K |
| `llama-3.3-70b` | 60K | 1M | 1M | 30 | 900 | 14.4K |
| `qwen-3-32b` | 60K | 1M | 1M | 30 | 900 | 14.4K |
| `qwen-3-235b-a22b-instruct-2507` | 60K | 1M | 1M | 30 | 900 | 14.4K |
| `zai-glm-4.7` | 60K | 1M | 1M | 10 | 100 | 100 |

## 11.5 Developer Tier Limits

Hourly and daily restrictions do not apply. Pay-as-you-go pricing.

| Model | TPM | RPM |
|-------|-----|-----|
| `gpt-oss-120b` | 1M | 1K |
| `llama3.1-8b` | 1M | 1K |
| `llama-3.3-70b` | 1M | 1K |
| `qwen-3-32b` | 1M | 1K |
| `qwen-3-235b-a22b-instruct-2507` | 1M | 1K |
| `zai-glm-4.7` | 500K | 500 |

## 11.6 Rate Limit Response Headers

| Header | Description |
|--------|-------------|
| `x-ratelimit-limit-requests-day` | Maximum requests per day |
| `x-ratelimit-limit-tokens-minute` | Maximum tokens per minute |
| `x-ratelimit-remaining-requests-day` | Remaining requests for current day |
| `x-ratelimit-remaining-tokens-minute` | Remaining tokens for current minute |
| `x-ratelimit-reset-requests-day` | Seconds until daily request limit resets |
| `x-ratelimit-reset-tokens-minute` | Seconds until per-minute token limit resets |

---

# 12. OpenAI Compatibility

## 12.1 Configuration

Use OpenAI client libraries by changing base URL and API key:

```python
import openai

client = openai.OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY")
)
```

```javascript
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.CEREBRAS_API_KEY,
    baseURL: "https://api.cerebras.ai/v1"
});
```

## 12.2 Passing Non-Standard Parameters

### OpenAI Client

Non-standard parameters must be passed through `extra_body`:

```python
response = client.chat.completions.create(
    model="zai-glm-4.7",
    messages=[...],
    extra_body={
        "disable_reasoning": False,
        "clear_thinking": False
    }
)
```

### Cerebras SDK

Non-standard parameters can be passed as regular parameters:

```python
response = client.chat.completions.create(
    model="zai-glm-4.7",
    messages=[...],
    disable_reasoning=False,
    clear_thinking=False
)
```

## 12.3 Developer Role Mapping (gpt-oss-120b)

For `gpt-oss-120b`, the `system` role maps to developer-level instructions. System messages are elevated above normal user instructions. The same prompt may yield different behavior compared to OpenAI's API.

## 12.4 Unsupported OpenAI Features

The following parameters return 400 error:
- `frequency_penalty`
- `logit_bias`
- `presence_penalty`

---

# 13. Error Handling

## 13.1 HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (invalid API key) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

## 13.2 Rate Limit Error (429)

When rate limits are exceeded, requests receive HTTP 429 status. Check rate limit headers for reset timing.

## 13.3 CloudFront Blocking

If requests are blocked by CloudFront, include `User-Agent` in headers.

---

# 14. Configuration Best Practices

## 14.1 Sampling Parameters

- Default: `temperature=1.0`, `top_p=1.0`
- For deterministic outputs: Adjust either `temperature` or `top_p`, not both
- For GLM with reasoning enabled: Avoid `temperature < 0.8` as it degrades output quality
- For more deterministic outputs with `temperature < 0.8`: Disable thinking

## 14.2 GLM 4.7 Recommended Settings

| Use Case | temperature | top_p | disable_reasoning | clear_thinking |
|----------|-------------|-------|-------------------|----------------|
| General | 1.0 | 0.95 | false | true |
| Instruction following | 0.8 | 0.95 | false | true |
| Agentic/coding workflows | 1.0 | 0.95 | false | false |
| Fast responses | 1.0 | 0.95 | true | - |

## 14.3 Prompting Best Practices for GLM 4.7

1. **Front-load instructions**: Place all rules and constraints at the beginning of system prompt
2. **Use clear, direct language**: Use MUST, REQUIRED, STRICTLY instead of "Please try to..."
3. **Specify default language**: Add "Always respond in English" to prevent language switching
4. **Use role prompts**: Assign clear roles for consistency
5. **Break down tasks**: Decompose complex work into substeps

---

# 15. Appendix

## 15.1 API Endpoint Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/models` | GET | List available models |

## 15.2 SDK Repository Links

- Python SDK: https://github.com/Cerebras/cerebras-cloud-sdk-python
- Node.js SDK: https://github.com/Cerebras/cerebras-cloud-sdk-node

## 15.3 Hugging Face Model Links

| Model ID | Hugging Face URL |
|----------|------------------|
| `llama3.1-8b` | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| `llama-3.3-70b` | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct |
| `gpt-oss-120b` | https://huggingface.co/openai/gpt-oss-120b |
| `qwen-3-32b` | https://huggingface.co/Qwen/Qwen3-32B |
| `qwen-3-235b-a22b-instruct-2507` | https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507 |
| `zai-glm-4.7` | https://huggingface.co/zai-org/GLM-4.7 |

## 15.4 Complete cURL Example

```bash
curl --location 'https://api.cerebras.ai/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer ${CEREBRAS_API_KEY}" \
--data '{
    "model": "llama-3.3-70b",
    "stream": false,
    "messages": [{"content": "Hello!", "role": "user"}],
    "temperature": 0,
    "max_completion_tokens": -1,
    "seed": 0,
    "top_p": 1
}'
```

## 15.5 Complete Python Example with Tool Calling

```python
import os
import json
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

def get_weather(location):
    return json.dumps({"location": location, "temperature": 22, "condition": "sunny"})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "strict": True,
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g., 'San Francisco, USA'"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Toronto?"}
]

response = client.chat.completions.create(
    model="zai-glm-4.7",
    messages=messages,
    tools=tools,
    parallel_tool_calls=True,
)

msg = response.choices[0].message

if msg.tool_calls:
    messages.append(msg.model_dump())
    
    for tool_call in msg.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = get_weather(args["location"])
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })
    
    final_response = client.chat.completions.create(
        model="zai-glm-4.7",
        messages=messages,
    )
    print(final_response.choices[0].message.content)
else:
    print(msg.content)
```

## 15.6 Glossary

| Term | Definition |
|------|------------|
| **Constrained Decoding** | Token-level schema enforcement ensuring valid outputs |
| **EOS** | End of Sequence token |
| **MoE** | Mixture of Experts architecture with sparse activation |
| **MSL** | Maximum Sequence Length |
| **Predicted Outputs** | Optimization for known output content |
| **Prompt Caching** | Automatic prefix caching for reduced latency |
| **RPM/RPH/RPD** | Requests per minute/hour/day |
| **Reasoning Tokens** | Intermediate thinking tokens generated before final response |
| **TPM/TPH/TPD** | Tokens per minute/hour/day |
| **TTFT** | Time to First Token |
| **TTL** | Time to Live (cache expiration) |
| **ZDR** | Zero Data Retention |
