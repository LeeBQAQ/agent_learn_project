import os
from typing import List

class SimpleEmbeddings:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension # 定义向量维度

    """一个简单的Embeddings类，用于测试和演示"""
    def embed_query(self, text):
        """嵌入查询"""
        return self._embed_text(text)

    def embed_documents(self, texts):
        """嵌入文档列表"""
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> List[float]:
        """简单的文本嵌入（基于字符频率）"""
        import hashlib
        # 使用文本的hash生成伪随机但确定的向量
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # 扩展到目标维度
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            # 归一化到 [-1, 1]
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            vector.append(value)

        return vector

def get_embeddings():
    """根据环境选择合适的Embeddings"""
    if os.getenv("EMBEDDING_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-ada-002")
        except ImportError:
            print("⚠️ langchain_openai 未安装，使用简单 Embeddings")
            raise ImportError("请安装 langchain_openai")
    return SimpleEmbeddings()

if __name__ == '__main__':
    if __name__ == '__main__':
        print('向量程序內部测试')
        embeddings = get_embeddings()
        query = embeddings.embed_query("hello world")
        print(query)
    else:
        print('向量程序外部测试')