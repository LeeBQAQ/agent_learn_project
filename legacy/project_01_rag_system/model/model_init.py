import os
from typing import Any

from datasets.io import json
from dotenv import load_dotenv
from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek
from module_01_embeddings.embedding import get_embeddings

class ChatDeepSeekFixReasoningContent(ChatDeepSeek):
    """修复deepseek思考内容不兼容问题"""
    def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
    ) -> dict:
        # 调用BaseChatOpenAI中的_get_request_payload
        payload = super(ChatDeepSeek, self)._get_request_payload(input_, stop=stop, **kwargs)
        input_messages = self._convert_input(input_).to_messages() or []
        for idx, message in enumerate(payload["messages"]):
            # 获取所有消息
            reasoning_content = input_messages[idx].additional_kwargs.get("reasoning_content")
            if reasoning_content and message["role"] == "assistant":
                # 添加reasoning_content字段
                message["reasoning_content"] = reasoning_content
            if message["role"] == "tool" and isinstance(message["content"], list):
                message["content"] = json.dumps(message["content"])
            elif message["role"] == "assistant" and isinstance(
                    message["content"], list
            ):
                text_parts = [
                    block.get("text", "")
                    for block in message["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                message["content"] = "".join(text_parts) if text_parts else ""
        return payload


load_dotenv()
MODEL_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("model")

# 初始化模型
print("Initializing model...")
model = ChatDeepSeekFixReasoningContent(
    model=MODEL,
    api_key=MODEL_API_KEY
)
# 初始化模型
print("Initializing no thinking model...")
no_thinking_model = ChatDeepSeekFixReasoningContent(
    model=MODEL,
    api_key=MODEL_API_KEY,
    extra_body={"thinking": {"type": "disabled"}}
)
print("Initializing embeddings model...")
embeddings_model = get_embeddings()