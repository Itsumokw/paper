"""
LLM Client - Handles all LLM interactions
"""
import json
import logging
import os
import threading
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import config


class LLMTruncatedOutputError(ValueError):
    """Raised when generation hits the token/context limit before completion."""


class LLMClient:
    """
    Unified LLM client interface
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.enable_thinking = enable_thinking if enable_thinking is not None else config.ENABLE_THINKING
        self.use_streaming = use_streaming if use_streaming is not None else config.USE_STREAMING
        self._usage_lock = threading.Lock()
        self._usage_records = []

        # Initialize OpenAI client with optional base_url
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            print(f"Using custom OpenAI base URL: {self.base_url}")

        if self.enable_thinking:
            print(f"Deep thinking mode enabled")

        # self.client = OpenAI(**client_kwargs)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,
        )

    def _estimate_tokens(self, text: str) -> int:
        # Lightweight, deterministic approximation used when streaming responses
        # do not expose provider token usage.
        if not text:
            return 0
        return max(1, len(str(text)) // 4)

    def _messages_token_estimate(self, messages: List[Dict[str, str]]) -> int:
        return sum(self._estimate_tokens(msg.get("content", "")) + 4 for msg in messages)

    def _trim_text_for_budget(self, text: str, budget_tokens: int) -> str:
        if self._estimate_tokens(text) <= budget_tokens:
            return text
        char_budget = max(256, budget_tokens * 4)
        if len(text) <= char_budget:
            return text
        head = max(128, char_budget // 2)
        tail = max(128, char_budget - head - 64)
        return f"{text[:head]}\n...[truncated for context budget]...\n{text[-tail:]}"

    def _fit_messages_and_max_tokens(self, messages: List[Dict[str, str]], requested_tokens: int):
        max_model_tokens = int(os.environ.get("SIMPLEMEM_MAX_MODEL_TOKENS", os.environ.get("VLLM_MAX_MODEL_LEN", "32768")))
        buffer_tokens = int(os.environ.get("SIMPLEMEM_TOKEN_GUARD_BUFFER", "1024"))
        min_output_tokens = int(os.environ.get("SIMPLEMEM_MIN_OUTPUT_TOKENS", "1024"))
        requested_tokens = max(min_output_tokens, int(requested_tokens))
        fitted = [dict(message) for message in messages]
        prompt_tokens = self._messages_token_estimate(fitted)
        available_tokens = max_model_tokens - prompt_tokens - buffer_tokens
        if available_tokens < min_output_tokens:
            content_indices = [
                i for i, msg in enumerate(fitted)
                if isinstance(msg.get("content"), str) and msg.get("content")
            ]
            if content_indices:
                overhead = sum(4 for _ in fitted)
                prompt_budget = max(256, max_model_tokens - min_output_tokens - buffer_tokens - overhead)
                per_message_budget = max(128, prompt_budget // len(content_indices))
                for i in content_indices:
                    fitted[i]["content"] = self._trim_text_for_budget(str(fitted[i].get("content", "")), per_message_budget)
                prompt_tokens = self._messages_token_estimate(fitted)
                available_tokens = max_model_tokens - prompt_tokens - buffer_tokens
        safe_tokens = max(1, min(requested_tokens, max(1, available_tokens)))
        return fitted, safe_tokens

    def _record_usage(
        self,
        *,
        messages: List[Dict[str, str]],
        content: str,
        elapsed: float,
        stream: bool,
        usage: Any = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is None:
            prompt_tokens = self._messages_token_estimate(messages)
        if completion_tokens is None:
            completion_tokens = self._estimate_tokens(content)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        record = {
            "model": self.model,
            "stream": stream,
            "success": success,
            "elapsed_seconds": elapsed,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "error": error,
        }
        with self._usage_lock:
            self._usage_records.append(record)

    def get_usage_summary(self) -> Dict[str, Any]:
        with self._usage_lock:
            records = list(self._usage_records)
        success = [r for r in records if r.get("success")]
        def avg(key: str) -> float:
            return sum(float(r.get(key) or 0) for r in success) / len(success) if success else 0.0
        return {
            "count": len(records),
            "success_count": len(success),
            "error_count": len(records) - len(success),
            "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in success),
            "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in success),
            "total_tokens": sum(int(r.get("total_tokens") or 0) for r in success),
            "avg_prompt_tokens_per_call": avg("prompt_tokens"),
            "avg_completion_tokens_per_call": avg("completion_tokens"),
            "avg_total_tokens_per_call": avg("total_tokens"),
            "avg_latency_seconds_per_call": avg("elapsed_seconds"),
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Standard chat completion with optional thinking mode and retry mechanism
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        max_output_tokens = max_tokens if max_tokens is not None else getattr(config, "MAX_OUTPUT_TOKENS", None)
        if max_output_tokens:
            messages, safe_max_tokens = self._fit_messages_and_max_tokens(messages, int(max_output_tokens))
            kwargs["messages"] = messages
            kwargs["max_tokens"] = safe_max_tokens

        kwargs["timeout"] = float(os.environ.get("SIMPLEMEM_OPENAI_TIMEOUT", "300"))

        if response_format:
            kwargs["response_format"] = response_format

        # Enable thinking mode if configured (for Qwen and compatible models only)
        # Only add enable_thinking parameter for Qwen API (identified by base_url)
        is_qwen_api = self.base_url and "dashscope.aliyuncs.com" in self.base_url
        
        if is_qwen_api:
            # Qwen API requires explicit enable_thinking parameter
            # - Streaming + thinking: enable_thinking=True
            # - Non-streaming: enable_thinking=False (required, not optional)
            # - JSON format: enable_thinking=False (incompatible with thinking mode)
            if self.use_streaming and self.enable_thinking and not response_format:
                kwargs["extra_body"] = {"enable_thinking": True}
            else:
                # Explicitly set to False for non-streaming calls or JSON format
                kwargs["extra_body"] = {"enable_thinking": False}
        # For OpenAI and other APIs, don't add extra_body parameters

        # Retry mechanism
        last_exception = None
        for attempt in range(max_retries):
            started = time.time()
            try:
                # Use streaming if configured
                if self.use_streaming:
                    kwargs["stream"] = True
                    content = self._handle_streaming_response(**kwargs)
                    self._record_usage(
                        messages=messages,
                        content=content,
                        elapsed=time.time() - started,
                        stream=True,
                    )
                    return content
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    choice = response.choices[0]
                    content = choice.message.content or ""
                    if getattr(choice, "finish_reason", None) == "length":
                        logging.warning(
                            "LLM response ended at the generation limit; "
                            "passing partial content to the JSON parser."
                        )
                    self._record_usage(
                        messages=messages,
                        content=content,
                        elapsed=time.time() - started,
                        stream=False,
                        usage=getattr(response, "usage", None),
                    )
                    return content
                
                # kwargs["stream"] = True
                # return self._handle_streaming_response(**kwargs)
                    
            except LLMTruncatedOutputError:
                raise
            except Exception as e:
                # print(e)
                last_exception = e
                self._record_usage(
                    messages=messages,
                    content="",
                    elapsed=time.time() - started,
                    stream=bool(kwargs.get("stream")),
                    success=False,
                    error=str(e),
                )
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM API call failed after {max_retries} attempts: {e}")
        
        # If all retries failed, raise the last exception
        raise last_exception

    def _handle_streaming_response(self, **kwargs) -> str:
        """
        Handle streaming response and collect content until the first complete
        top-level JSON value closes. This avoids waiting for extra text after
        the usable JSON has already been generated.
        """
        full_content = []
        stream = self.client.chat.completions.create(**kwargs)
        stopped_on_json = False
        finish_reason = None

        for chunk in stream:
            # fix list index out of range
            if len(chunk.choices) > 0:
                choice = chunk.choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                content = choice.delta.content
                if content is None:
                    continue
                full_content.append(content)
                current_text = ''.join(full_content)
                json_end = self._find_first_complete_json_end(current_text)
                if json_end is not None:
                    stopped_on_json = True
                    try:
                        stream.close()
                    except Exception:
                        pass
                    return current_text[:json_end + 1]

        text = ''.join(full_content)
        if finish_reason == "length":
            logging.warning(
                "Streaming LLM response ended at the model generation limit; "
                "passing partial content to the JSON parser."
            )
            return text
        if not stopped_on_json:
            return text
        return text

    def _find_first_complete_json_end(self, text: str) -> Optional[int]:
        """
        Return the end index of the first balanced top-level JSON object/array
        in text, or None if it has not closed yet.
        """
        starts = []
        for i, char in enumerate(text):
            if char in ['[', '{']:
                starts.append((i, char))

        for start_idx, start_char in starts:
            end_idx = self._find_balanced_json_end(text, start_idx, start_char)
            if end_idx is None:
                if self._looks_like_json_start(text, start_idx, start_char):
                    return None
                continue

            candidate = text[start_idx:end_idx + 1]
            try:
                json.loads(candidate)
                return end_idx
            except json.JSONDecodeError:
                cleaned = self._clean_json_string(candidate)
                try:
                    json.loads(cleaned)
                    return end_idx
                except json.JSONDecodeError:
                    continue

        return None

    def _looks_like_json_start(self, text: str, start_idx: int, start_char: str) -> bool:
        """
        Heuristic to avoid treating a complete child object inside an incomplete
        top-level array/object as the final streamed response.
        """
        next_idx = start_idx + 1
        while next_idx < len(text) and text[next_idx].isspace():
            next_idx += 1

        if next_idx >= len(text):
            return True

        next_char = text[next_idx]
        if start_char == '[':
            return next_char in '{["-0123456789tfn]'
        return next_char in '"}'

    def _find_balanced_json_end(self, text: str, start_idx: int, start_char: str) -> Optional[int]:
        """
        Return the closing index for a JSON value that starts at start_idx.
        """
        end_char = '}' if start_char == '{' else ']'
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    return i

        return None

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response with robust parsing
        Supports multiple formats:
        1. Pure JSON
        2. ```json ... ```
        3. ``` ... ``` (generic code block)
        4. JSON embedded in text with common prefixes
        5. Multiple JSON objects (returns first valid one)
        """
        if not text or not text.strip():
            raise ValueError("Empty response received")

        text = text.strip()

        # Remove common LLM prefixes/suffixes
        common_prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON is:",
            "JSON:",
            "Result:",
            "Output:",
            "Answer:",
        ]
        for prefix in common_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from ```json ... ``` block
        if "```json" in text.lower():
            # Case insensitive search for ```json
            start_marker = "```json"
            start_idx = text.lower().find(start_marker)
            if start_idx != -1:
                start = start_idx + len(start_marker)
                # Find the closing ```
                end = text.find("```", start)
                if end != -1:
                    json_str = text[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to clean up common issues
                        json_str = self._clean_json_string(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # Try extracting from generic ``` ... ``` code block
        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up
                    json_str = self._clean_json_string(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        # Try finding balanced JSON from the first JSON-looking structure.
        # If an array starts first but never closes, do not accidentally parse
        # the first object inside the incomplete array as a complete response.
        starts = []
        for start_char in ['[', '{']:
            start_idx = text.find(start_char)
            if start_idx != -1:
                starts.append((start_idx, start_char))
        for start_idx, start_char in sorted(starts):
            result = self._extract_balanced_json(text, start_char, start_idx=start_idx)
            if result is not None:
                return result
            if start_char == '[':
                raise ValueError("Found JSON array start but no balanced closing ']'")

        # Last resort: try to find any JSON-like structure and clean it
        for start_char in ['[', '{']:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Extract a large chunk and try to parse
                chunk = text[start_idx:]
                cleaned = self._clean_json_string(chunk)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to extract valid JSON from response. First 300 chars: {text[:300]}...")

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common issues in JSON strings from LLM output
        """
        # Remove trailing commas before } or ]
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Remove comments (// and /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        return json_str.strip()

    def _extract_balanced_json(self, text: str, start_char: str, start_idx: Optional[int] = None) -> Any:
        """
        Extract a balanced JSON object or array starting with start_char
        """
        end_char = '}' if start_char == '{' else ']'
        if start_idx is None:
            start_idx = text.find(start_char)

        if start_idx == -1:
            return None

        # Track depth to find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Handle strings (don't count brackets inside strings)
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            # Count depth
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try cleaning and parsing again
                        cleaned = self._clean_json_string(json_str)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            # Continue searching for next occurrence
                            break

        return None
