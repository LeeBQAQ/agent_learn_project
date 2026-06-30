import os

# 国内用户自动使用 HuggingFace 镜像，必须在导入 langchain_huggingface 之前设置
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings


def get_local_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embeddings

def get_embeddings():
    """根据环境选择合适的Embeddings"""
    if os.getenv("EMBEDDING_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-ada-002")
        except ImportError:
            print("⚠️ langchain_openai 未安装，使用简单 Embeddings")
            raise ImportError("请安装 langchain_openai")
    print("⚠️ 未设置 EMBEDDING_API_KEY，使用简单 Embeddings")
    return get_local_embeddings()