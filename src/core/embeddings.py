import os

if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings


def _get_local_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings():
    """根据环境选择合适的 Embeddings 模型"""
    if os.getenv("EMBEDDING_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model="text-embedding-ada-002")
        except ImportError:
            raise ImportError("请安装 langchain_openai")
    return _get_local_embeddings()
