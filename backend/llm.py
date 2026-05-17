import json
import asyncio
from .config import load_config


class LLMClient:
    """MiniMax OpenAI-compatible LLM client wrapper."""

    async def chat_async(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
        """Async version using httpx for concurrent judge calls."""
        last_error = None
        config = load_config()
        max_retries = config["llm_max_retries"]

        for attempt in range(max_retries + 1):
            try:
                return await self._chat_async_once(system_prompt, user_prompt, temperature)
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries or not self._is_retryable_error(exc):
                    break
                await asyncio.sleep(0.6 * (attempt + 1))

        raise last_error

    async def _chat_async_once(self, system_prompt: str, user_prompt: str, temperature: float) -> dict:
        """Send one async request to the OpenAI-compatible API."""
        import httpx

        config = load_config()
        api_key = config["minimax_api_key"]
        if not api_key:
            raise Exception("MiniMax API Key 未配置，请在页面的模型配置中填写。")

        async with httpx.AsyncClient(base_url=config["minimax_base_url"], timeout=120.0) as client:
            response = await client.post(
                "/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                },
            )
            try:
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                raise Exception(f"API HTTP Error {response.status_code}: {response.text[:500]}") from exc
            except json.JSONDecodeError as exc:
                raise Exception(f"API returned non-JSON response: {response.text[:500]}") from exc

            if "error" in data:
                raise Exception(f"API Error: {data['error']}")

            # Extract token usage
            usage = data.get("usage", {})
            token_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

            content = data["choices"][0]["message"].get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            if not content:
                raise Exception(f"Empty response from API. Full response: {data}")
            # MiniMax returns thinking blocks mixed in content - extract actual text
            text_content = self._extract_text_from_content(content)
            result = self._parse_json_with_fallback(text_content)
            result["_token_info"] = token_info
            return result

    def _is_retryable_error(self, exc: Exception) -> bool:
        message = str(exc)
        retryable_markers = [
            "Failed to parse JSON",
            "API returned non-JSON response",
            "Empty response from API",
            "API HTTP Error 408",
            "API HTTP Error 409",
            "API HTTP Error 425",
            "API HTTP Error 429",
            "API HTTP Error 500",
            "API HTTP Error 502",
            "API HTTP Error 503",
            "API HTTP Error 504",
            "timed out",
            "ReadTimeout",
            "ConnectTimeout",
            "RemoteProtocolError",
            "ConnectError",
        ]
        return any(marker in message for marker in retryable_markers)

    def _parse_json_with_fallback(self, text: str) -> dict:
        """Parse JSON with fallback for malformed responses from LLM."""
        import re
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to find and extract JSON object
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                json_text = match.group()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    repaired = self._escape_unescaped_inner_quotes(json_text)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass
            raise Exception(f"Failed to parse JSON: {e}\nText: {text[:300]}")

    def _escape_unescaped_inner_quotes(self, text: str) -> str:
        """Repair common LLM JSON where quotes inside string values are not escaped."""
        repaired = []
        in_string = False
        escaped = False

        for index, char in enumerate(text):
            if char == "\\" and in_string:
                repaired.append(char)
                escaped = not escaped
                continue

            if char == '"' and not escaped:
                if in_string:
                    next_non_space = self._next_non_space(text, index + 1)
                    if next_non_space in {":", ",", "}", "]"}:
                        in_string = False
                        repaired.append(char)
                    else:
                        repaired.append('\\"')
                else:
                    in_string = True
                    repaired.append(char)
            else:
                repaired.append(char)
                escaped = False

        return "".join(repaired)

    def _next_non_space(self, text: str, start: int) -> str:
        for char in text[start:]:
            if not char.isspace():
                return char
        return ""

    def _extract_text_from_content(self, content: str) -> str:
        """Remove thinking blocks and code block wrappers from MiniMax response content."""
        import re
        # Remove <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Remove ```json ... ``` code block wrappers
        cleaned = re.sub(r'```json\s*(.*?)\s*```', r'\1', cleaned, flags=re.DOTALL)
        # Also handle plain ``` ... ```
        cleaned = re.sub(r'```\s*(.*?)\s*```', r'\1', cleaned, flags=re.DOTALL)
        return cleaned.strip()


llm_client = LLMClient()
