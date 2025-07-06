import chromadb
from chromadb.config import Settings
from openai import OpenAI


class FinancialSituationMemory:
    def __init__(self, name, config):
        import os
        from ...config_manager import ConfigManager
        
        # 获取配置管理器实例
        config_manager = ConfigManager()
        
        # 获取嵌入模型配置
        embedding_config = config_manager.get_embedding_config()
        
        # 保存嵌入向量维度
        self.embedding_dimensions = embedding_config.get('dimensions', 1536)
        
        # 检查嵌入配置是否有效
        if not embedding_config['api_key']:
            print(f"警告：未找到嵌入模型API密钥（{embedding_config['provider']}），嵌入功能将被禁用")
            self.embedding_enabled = False
            self.embedding = None
            # 创建一个虚拟客户端用于其他功能
            self.client = OpenAI(
                api_key=config.get("api_key", "dummy"),
                base_url=config.get("backend_url", "https://api.openai.com/v1")
            )
        else:
            # 使用嵌入配置创建客户端
            self.client = OpenAI(
                api_key=embedding_config['api_key'],
                base_url=embedding_config['base_url']
            )
            self.embedding = embedding_config['model']
            self.embedding_enabled = True
            
            print(f"嵌入模型配置: {embedding_config['provider']}/{embedding_config['model']} (维度: {self.embedding_dimensions})")
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get embedding for a text"""
        if not self.embedding_enabled:
            # 如果嵌入功能被禁用，返回一个虚拟的嵌入向量
            import hashlib
            # 使用文本的哈希值生成一个固定长度的向量
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # 将哈希值转换为指定维度的向量
            vector = []
            for i in range(0, len(hash_hex), 2):
                vector.append(int(hash_hex[i:i+2], 16) / 255.0)
            # 填充到指定维度
            while len(vector) < self.embedding_dimensions:
                vector.extend(vector[:min(len(vector), self.embedding_dimensions - len(vector))])
            return vector[:self.embedding_dimensions]
        
        response = self.client.embeddings.create(
            model=self.embedding, 
            input=text,
            dimensions=self.embedding_dimensions
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
