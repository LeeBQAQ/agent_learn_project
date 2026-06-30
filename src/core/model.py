import os
import json
from typing import Any
from dotenv import load_dotenv
from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek

load_dotenv()


class _ChatDeepSeekFixReasoningContent(ChatDeepSeek):
    """修复 deepseek 思考内容不兼容问题"""

    def _get_request_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict:
        payload = super(ChatDeepSeek, self)._get_request_payload(input_, stop=stop, **kwargs)
        input_messages = self._convert_input(input_).to_messages() or []
        for idx, message in enumerate(payload["messages"]):
            reasoning_content = input_messages[idx].additional_kwargs.get("reasoning_content")
            if reasoning_content and message["role"] == "assistant":
                message["reasoning_content"] = reasoning_content
            if message["role"] == "tool" and isinstance(message["content"], list):
                message["content"] = json.dumps(message["content"])
            elif message["role"] == "assistant" and isinstance(message["content"], list):
                text_parts = [
                    block.get("text", "")
                    for block in message["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                message["content"] = "".join(text_parts) if text_parts else ""
        return payload


MODEL_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("MODEL", "deepseek-chat")

model = _ChatDeepSeekFixReasoningContent(model=MODEL_NAME, api_key=MODEL_API_KEY)
no_thinking_model = _ChatDeepSeekFixReasoningContent(
    model=MODEL_NAME, api_key=MODEL_API_KEY, extra_body={"thinking": {"type": "disabled"}}
)
