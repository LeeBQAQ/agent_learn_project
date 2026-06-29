import os
from typing import Any

from datasets.io import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI


class ChatDeepSeekFixReasoningContent(ChatDeepSeek):
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
GROQ_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("model")

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 GROQ_API_KEY\n"
        "访问 https://console.groq.com/keys 获取免费密钥"
    )

# 初始化模型
model = ChatDeepSeekFixReasoningContent(model=MODEL,
                                        api_key=GROQ_API_KEY
                                        )
# 初始化模型
no_thinking_model = ChatDeepSeekFixReasoningContent(
    model=MODEL,
    api_key=GROQ_API_KEY,
    extra_body={"thinking": {"type": "disabled"}}
)

# model =  ChatOpenAI(
#     api_key=QWEN_API_KEY,
#     base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",
#     model= QWEN_MODEL,
#     model_kwargs={"enable_thinking": False}
#     # other params...
# )
