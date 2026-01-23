"""
远程 Embedding 服务客户端
连接 192.168.1.27:8100 的 Qwen3-VL-Embedding 服务
"""
import requests
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embedding 服务客户端"""

    def __init__(
        self,
        host: str = "192.168.1.27",
        port: int = 8100,
        timeout: int = 120
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._connected = False

    def health_check(self) -> bool:
        """健康检查"""
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            if resp.status_code == 200:
                self._connected = True
                return True
            return False
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def embed_text(
        self,
        text: str,
        instruction: Optional[str] = None
    ) -> List[float]:
        """
        单条文本向量化

        Args:
            text: 待向量化的文本
            instruction: 可选的指令前缀

        Returns:
            768维向量列表
        """
        payload = {"text": text}
        if instruction:
            payload["instruction"] = instruction

        try:
            resp = requests.post(
                f"{self.base_url}/embed/text",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except requests.exceptions.Timeout:
            logger.error(f"向量化超时 (>{self.timeout}s)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"向量化请求失败: {e}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None
    ) -> List[List[float]]:
        """
        批量文本向量化

        Args:
            texts: 文本列表
            instruction: 可选的指令前缀

        Returns:
            向量列表
        """
        payload = {"texts": texts}
        if instruction:
            payload["instruction"] = instruction

        try:
            resp = requests.post(
                f"{self.base_url}/embed/batch",
                json=payload,
                timeout=self.timeout * 2
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            raise

    def embed_patent(
        self,
        title: str,
        abstract: str = "",
        claims: str = "",
        description: str = ""
    ) -> Dict[str, List[float]]:
        """
        专利多字段向量化

        Args:
            title: 发明名称
            abstract: 摘要
            claims: 权利要求
            description: 说明书（可选）

        Returns:
            包含各字段向量的字典:
            {
                "title_vector": [...],
                "abstract_vector": [...],
                "claims_vector": [...],
                "description_vector": [...]  # 如果提供了description
            }
        """
        payload = {
            "title": title,
            "abstract": abstract,
            "claims": claims,
        }
        if description:
            payload["description"] = description

        try:
            resp = requests.post(
                f"{self.base_url}/embed/patent",
                json=payload,
                timeout=self.timeout * 3
            )
            resp.raise_for_status()
            return resp.json()["vectors"]
        except Exception as e:
            logger.error(f"专利向量化失败: {e}")
            raise

    def get_server_info(self) -> Dict:
        """获取服务器信息"""
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"获取服务器信息失败: {e}")
            return {"status": "error", "message": str(e)}


# 全局客户端实例（使用默认配置）
embedding_client = EmbeddingClient()


# 便捷函数
def get_embedding_client(host: str = "192.168.1.27", port: int = 8100) -> EmbeddingClient:
    """获取 Embedding 客户端实例"""
    return EmbeddingClient(host=host, port=port)


if __name__ == "__main__":
    # 测试连接
    logging.basicConfig(level=logging.INFO)

    print("测试 Embedding 服务连接...")
    client = EmbeddingClient()

    if client.health_check():
        print("✅ 服务连接正常")
        info = client.get_server_info()
        print(f"服务器信息: {info}")

        # 测试向量化
        print("\n测试文本向量化...")
        test_text = "A method for image recognition using deep learning"
        try:
            embedding = client.embed_text(test_text)
            print(f"向量维度: {len(embedding)}")
            print(f"前5位: {embedding[:5]}")
        except Exception as e:
            print(f"❌ 向量化测试失败: {e}")
    else:
        print("❌ 无法连接到 Embedding 服务")
        print("请确保 192.168.1.27:8100 服务已启动")
