import os
import jieba
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 配置 Embedding 模型 (以 SiliconFlow 为例，兼容 OpenAI 格式)
os.environ["OPENAI_API_KEY"] = "your_siliconflow_api_key"
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    base_url="https://api.siliconflow.cn/v1",
)


def ingest_data():
    # 1. 加载 Markdown 数据
    loader = DirectoryLoader("./data", glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    # 2. 按 Markdown 标题层级切分 (保留上下文结构，对安全文档极度友好)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = []
    for doc in docs:
        splits.extend(splitter.split_text(doc.page_content))

    # 3. 初始化 Chroma 向量库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="security_kb",
        persist_directory="./chroma_db"
    )

    # 4. 准备 BM25 检索器 (需要自定义中文分词)
    # 将切分后的文本用 jieba 分词，存入本地 json 或直接内存使用
    print(f"成功入库 {len(splits)} 个文档块！")


if __name__ == "__main__":
    ingest_data()