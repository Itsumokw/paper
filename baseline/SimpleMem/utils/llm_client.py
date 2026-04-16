"""
LLM Client - Handles all LLM interactions
"""
import json
import threading
from typing import List, Dict, Any, Optional
from openai import OpenAI
import config


class LLMClient:
    """
    Unified LLM client interface
    """
    _local_backend_lock = threading.Lock()
    _local_model_lock = threading.Lock()
    _local_model = None
    _local_tokenizer = None
    _local_model_name = None
    _local_device = "cpu"

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
        self.is_local_transformers = bool(self.base_url and self.base_url.startswith("local://"))

        if self.is_local_transformers:
            print(f"Using local Transformers backend: {self.model}")
            self.client = None
            self._ensure_local_backend_loaded()
        else:
            # Initialize OpenAI client with optional base_url
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                print(f"Using custom OpenAI base URL: {self.base_url}")

            if self.enable_thinking:
                print(f"Deep thinking mode enabled")

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> str:
        """
        Standard chat completion with optional thinking mode and retry mechanism
        """
        if self.is_local_transformers:
            return self._local_chat_completion(
                messages=messages,
                temperature=temperature,
                max_retries=max_retries
            )

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

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
            try:
                # Use streaming if configured
                if self.use_streaming:
                    kwargs["stream"] = True
                    return self._handle_streaming_response(**kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    if not hasattr(response, "choices"):
                        response_type = type(response).__name__
                        raise TypeError(
                            f"Non-OpenAI-compatible response type returned: {response_type}. "
                            f"Current base_url={self.base_url}. "
                            f"If you are using a relay endpoint, check whether the base_url should end with '/v1'."
                        )
                    return response.choices[0].message.content
                
                # kwargs["stream"] = True
                # return self._handle_streaming_response(**kwargs)
                    
            except Exception as e:
                # print(e)
                last_exception = e
                if attempt < max_retries - 1:
                    import time
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM API call failed after {max_retries} attempts: {e}")
        
        # If all retries failed, raise the last exception
        raise last_exception

    def _ensure_local_backend_loaded(self) -> None:
        if (
            LLMClient._local_model is not None
            and LLMClient._local_tokenizer is not None
            and LLMClient._local_model_name == self.model
        ):
            return

        with LLMClient._local_model_lock:
            if (
                LLMClient._local_model is not None
                and LLMClient._local_tokenizer is not None
                and LLMClient._local_model_name == self.model
            ):
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            device_map = "auto" if device == "cuda" else None

            print(f"Loading local causal LM: {self.model}")
            print(f"Local generation device: {device}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                trust_remote_code=True
            )

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
            }
            if device_map is not None:
                model_kwargs["device_map"] = device_map

            model = AutoModelForCausalLM.from_pretrained(
                self.model,
                **model_kwargs
            )
            model.eval()

            LLMClient._local_model = model
            LLMClient._local_tokenizer = tokenizer
            LLMClient._local_model_name = self.model
            LLMClient._local_device = device

            print("Local causal LM loaded successfully")

    def _local_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_retries: int
    ) -> str:
        import torch

        self._ensure_local_backend_loaded()

        tokenizer = LLMClient._local_tokenizer
        model = LLMClient._local_model
        local_device = LLMClient._local_device

        last_exception = None
        for attempt in range(max_retries):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                model_device = next(model.parameters()).device
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(model_device) for key, value in inputs.items()}

                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": 512,
                    "pad_token_id": tokenizer.eos_token_id,
                }

                if temperature > 0:
                    generation_kwargs["do_sample"] = True
                    generation_kwargs["temperature"] = max(temperature, 0.1)
                    generation_kwargs["top_p"] = 0.9
                else:
                    generation_kwargs["do_sample"] = False

                with LLMClient._local_backend_lock:
                    with torch.inference_mode():
                        outputs = model.generate(**generation_kwargs)

                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                if not text:
                    raise ValueError("Local model returned empty text")
                return text
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    import time
                    wait_time = (2 ** attempt)
                    print(f"Local LLM call failed (attempt {attempt + 1}/{max_retries}) on {local_device}: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Local LLM call failed after {max_retries} attempts: {e}")

        raise last_exception

    def _handle_streaming_response(self, **kwargs) -> str:
        """
        Handle streaming response and collect full content
        """
        full_content = []
        stream = self.client.chat.completions.create(**kwargs)

        # for chunk in stream:
        #     if chunk.choices is not None:
        #         print(chunk.choices[0].delta.content)
        
        # print('---------')

        for chunk in stream:
            # print(chunk)
            # fix list index out of range
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_content.append(content)
                # print(full_content)
                # Optional: print streaming content in real-time
                # print(content, end='', flush=True)
        # print(full_content)
        print()
        return ''.join(full_content)

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

        # Try finding balanced JSON object/array by scanning for { or [
        for start_char in ['{', '[']:
            result = self._extract_balanced_json(text, start_char)
            if result is not None:
                return result

        # Last resort: try to find any JSON-like structure and clean it
        for start_char in ['{', '[']:
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

    def _extract_balanced_json(self, text: str, start_char: str) -> Any:
        """
        Extract a balanced JSON object or array starting with start_char
        """
        end_char = '}' if start_char == '{' else ']'
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
