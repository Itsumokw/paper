"""
Answer Generator - Final synthesis from retrieved contexts

Section 3.3: Intent-Aware Retrieval Planning
Generates answers from the merged context C_q after multi-view retrieval
"""
from typing import List
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
import config
import re
from datetime import datetime


class AnswerGenerator:
    """
    Answer Generator - Synthesis from retrieved memory units (Section 3.3)

    Generates answers from C_q = R_sem ∪ R_lex ∪ R_sym
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_answer(self, query: str, contexts: List[MemoryEntry]) -> str:
        """
        Generate answer

        Args:
        - query: User question
        - contexts: List of retrieved relevant MemoryEntry

        Returns:
        - Generated answer (concise phrase)
        """
        if not contexts:
            return "No relevant information found"

        # Build context string
        context_str = self._format_contexts(contexts)

        # Build prompt
        prompt = self._build_answer_prompt(query, context_str)

        # Call LLM to generate answer
        messages = [
            {
                "role": "system",
                "content": "You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                # Parse JSON response
                result = self.llm_client.extract_json(response)
                # Return the answer from JSON
                return self._normalize_answer(result.get("answer", response.strip()))

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to parse JSON response after {max_retries} attempts: {e}")
                    # Fallback to raw response
                    if 'response' in locals():
                        return self._normalize_answer(response.strip())
                    else:
                        return "Failed to generate answer"

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """
        Format contexts to readable text
        """
        formatted = []
        for i, entry in enumerate(contexts, 1):
            parts = [f"[Context {i}]"]
            parts.append(f"Content: {entry.lossless_restatement}")

            if entry.timestamp:
                parts.append(f"Time: {entry.timestamp}")

            if entry.location:
                parts.append(f"Location: {entry.location}")

            if entry.persons:
                parts.append(f"Persons: {', '.join(entry.persons)}")

            if entry.entities:
                parts.append(f"Related Entities: {', '.join(entry.entities)}")

            if entry.topic:
                parts.append(f"Topic: {entry.topic}")

            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)

    def _build_answer_prompt(self, query: str, context_str: str) -> str:
        """
        Build answer generation prompt
        """
        return f"""
You are an intelligent memory assistant tasked with answering LoCoMo questions
from retrieved conversation memories.

User Question: {query}

Relevant Context:
{context_str}

Requirements:
1. Answer based ONLY on the provided context.
2. Output one precise, concise answer; usually less than 5-6 words.
3. Do not include reasoning, explanations, citations, or copied context.
4. If the answer is a date, convert ISO dates such as 2023-05-01 to natural text such as "1 May 2023".
5. If a memory contains relative time such as "last year" or "two months ago", resolve it using that memory's timestamp.
6. If multiple memories conflict, prefer the most recent relevant memory.
7. Return only valid JSON with a single key: "answer".

Output Format:
```json
{{
  "answer": "Concise answer in a short phrase"
}}
```

Example:
Question: "When will they meet?"
Context: "Alice suggested meeting Bob at 2025-11-16T14:00:00..."

Output:
```json
{{
  "answer": "16 November 2025 at 2:00 PM"
}}
```

Now answer the question. Return ONLY the JSON, no other text.
"""

    def _normalize_answer(self, answer: str) -> str:
        """
        Keep metric-facing answers concise and normalize common date formats.
        """
        if answer is None:
            return ""
        text = str(answer).strip()
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
        text = re.sub(r"^(answer|final answer)\s*:\s*", "", text, flags=re.I).strip()
        if text.startswith("{"):
            try:
                import json

                parsed = json.loads(text)
                if isinstance(parsed, dict) and "answer" in parsed:
                    text = str(parsed["answer"]).strip()
            except json.JSONDecodeError:
                pass

        def repl(match: re.Match) -> str:
            raw = match.group(0)
            try:
                dt = datetime.strptime(raw[:10], "%Y-%m-%d")
            except ValueError:
                return raw
            return f"{dt.day} {dt.strftime('%B')} {dt.year}"

        text = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[T ][0-9:]{5,8})?\b", repl, text)
        return text.strip()
