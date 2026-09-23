from typing import List, TypedDict, Annotated
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langgraph.graph import END, StateGraph, START
import os



# 1. 定义 Agent 状态
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retry_count: int


# 初始化 LLM 和 检索器
os.environ["OPENROUTER_API_KEY"] = "your_openrouter_key"
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # 或者选免费的 qwen 模型
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

embeddings = OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-8B", base_url="https://api.siliconflow.cn/v1")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="security_kb")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# BM25 检索器 (假设你在 ingest 时保存了分词后的文本)
# 这里简化处理，实际应从数据库或内存加载
bm25_retriever = BM25Retriever.from_documents(vectorstore.get())
bm25_retriever.k = 5

# RRF 混合检索 (EnsembleRetriever 默认使用 RRF 算法融合)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 向量占60%，BM25占40% (安全领域精确匹配CVE/命令很重要)
)


# 2. 定义节点 (Nodes)
def retrieve(state):
    """检索节点：执行 RRF 混合检索"""
    question = state["question"]
    documents = ensemble_retriever.invoke(question)
    return {"documents": [doc.page_content for doc in documents]}


def rewrite_query(state):
    """查询重写节点：利用 LLM 将模糊的安全问题转化为精确的检索词"""
    question = state["question"]
    prompt = f"""你是一个网络安全专家。请将用户的模糊问题重写为适合在安全知识库中检索的精确查询。
    保留关键实体（如 CVE 编号、漏洞名、工具名）。
    原始问题: {question}
    重写后的查询:"""
    response = llm.invoke(prompt)
    return {"question": response.content}


def generate(state):
    """生成节点：基于检索到的上下文回答问题"""
    question = state["question"]
    documents = "\n\n---\n\n".join(state["documents"])

    prompt = ChatPromptTemplate.from_template("""
    你是一个资深的网络安全专家。请仅根据以下提供的知识库上下文回答用户的问题。
    如果上下文中没有答案，请明确告知“知识库中未找到相关安全信息”，不要编造（防止幻觉）。
    如果涉及代码或命令，请使用 Markdown 代码块格式。

    上下文:
    {documents}

    问题: {question}
    """)
    chain = prompt | llm
    generation = chain.invoke({"documents": documents, "question": question})
    return {"generation": generation.content}


# 3. 定义条件边 (Conditional Edges)
def decide_to_generate(state):
    """判断检索到的文档是否足够，决定是直接生成还是重写查询"""
    # 简单策略：如果检索到的文档总字数太少，说明没搜到，去重写查询
    total_length = sum(len(doc) for doc in state["documents"])
    if total_length < 200 and state.get("retry_count", 0) < 2:
        return "rewrite"
    return "generate"


# 4. 构建 LangGraph 图
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate", generate)

# 添加边
workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    decide_to_generate,
    {"rewrite": "rewrite_query", "generate": "generate"}
)
workflow.add_edge("rewrite_query", "retrieve")  # 重写后再次检索
workflow.add_edge("generate", END)

# 编译图
app = workflow.compile()