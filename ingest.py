"""ingest.py — 无 langchain-community 依赖版本"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import *


def load_markdown_docs(kb_dir: str = "./data/01_OWASP防御速查表") -> list[Document]:
    """轻量级 Markdown 加载器，替代 UnstructuredMarkdownLoader"""
    docs = []
    kb_path = Path(kb_dir)
    for md_file in sorted(kb_path.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        metadata = {
            "source": str(md_file),
            "filename": md_file.name,
        }
        docs.append(Document(page_content=content, metadata=metadata))
    print(f"📄 加载了 {len(docs)} 个 Markdown 文档")
    return docs


def load_and_split(kb_dir: str = "./data/01_OWASP防御速查表"):
    # 1. 加载（纯标准库，无需 community/unstructured）
    docs = load_markdown_docs(kb_dir)

    # 2. 按 Markdown 标题层级切分
    headers_to_split_on = [
        ("#", "一级标题"),
        ("##", "二级标题"),
        ("###", "三级标题"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_chunks = []
    for doc in docs:
        splits = md_splitter.split_text(doc.page_content)
        # 将原始文件元数据合并到每个 chunk
        for split in splits:
            split.metadata.update(doc.metadata)
        md_chunks.extend(splits)

    # 3. 二次递归切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    final_chunks = text_splitter.split_documents(md_chunks)
    print(f"✂️  切分为 {len(final_chunks)} 个 chunks")
    return final_chunks


def build_vectordb(chunks, persist_dir: str = CHROMA_PERSIST_DIR):
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
    # ✅ 使用 langchain-chroma 独立包
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    print(f"✅ 向量库已构建，共 {vectordb._collection.count()} 条记录")
    return vectordb


if __name__ == "__main__":
    chunks = load_and_split()
    build_vectordb(chunks)