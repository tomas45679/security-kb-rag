import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
# 1. 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-8b",  # 替换为你实际调用的 Qwen Embedding 模型 ID
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),  # 或直接填入你的 Key
    check_embedding_ctx_length=False
)


def test_embedding():
    print("=" * 50)
    print("🚀 开始测试 Qwen Embedding 模型...")
    print("=" * 50)

    # 2. 测试单条文本嵌入
    try:
        single_vector = embeddings.embed_query("人工智能正在改变世界")
        print(f"✅ API 连通性: 成功")
        print(f"📏 向量维度: {len(single_vector)}")
        print(f"🔢 前5个值: {single_vector[:5]}")
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return

    # 3. 测试批量嵌入 & 语义相似度
    texts = [
        "今天天气真好，适合出去散步",  # A
        "阳光明媚，非常适合户外活动",  # B (与A语义相似)
        "量子力学是物理学的一个分支",  # C (与A无关)
    ]

    vectors = embeddings.embed_documents(texts)

    # 计算余弦相似度
    sim_ab = np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
    sim_ac = np.dot(vectors[0], vectors[2]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[2]))

    print(f"\n📊 语义相似度测试:")
    print(f"   A vs B (相似句): {sim_ab:.4f}")
    print(f"   A vs C (无关句): {sim_ac:.4f}")

    if sim_ab > sim_ac:
        print(f"✅ 语义区分能力: 正常 (相似句得分 > 无关句)")
    else:
        print(f"⚠️ 语义区分能力: 异常 (请检查模型是否正确)")

    print("=" * 50)
    print("🎉 测试完成！模型可正常使用于 RAG 流程")
    print("=" * 50)


if __name__ == "__main__":
    test_embedding()