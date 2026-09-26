"""
LangGraph RAG 图：
START → retrieve → grade_relevance → [相关] → generate → END
                                   → [不相关] → rewrite_query → retrieve (循环)
"""
from typing import TypedDict, List, Annotated
from operator import add

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from config import *


# ============ State 定义 ============
class RAGState(TypedDict):
    question: str                    # 用户原始问题
    rewritten_question: str          # 改写后的查询（可能多轮）
    documents: List[str]             # 检索到的文档片段
    answer: str                      # 最终回答
    relevance_score: str             # "yes" / "no"
    retry_count: int                 # 重试次数


# ============ 初始化组件 ============
def get_llm():
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )

def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False
    )

def get_vectordb():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ============ 节点函数 ============
def retrieve(state: RAGState) -> dict:
    """检索节点：从向量库中检索相关文档"""
    print(f"[retrieve] 查询: {state.get('rewritten_question') or state['question']}", flush=True)
    vectordb = get_vectordb()
    query = state.get("rewritten_question") or state["question"]
    docs = vectordb.similarity_search(query, k=TOP_K)
    print(f"[retrieve] 检索到 {len(docs)} 个文档", flush=True)
    # if docs:
    #     print(f"[retrieve] 第一个文档片段: {docs[0].page_content[:80]}...", flush=True)
    return {"documents": [doc.page_content for doc in docs]}


def grade_relevance(state: RAGState) -> dict:
    """相关性评估节点：判断检索结果是否与问题相关"""
    print(f"[grade] 开始评估相关性，retry_count={state.get('retry_count',0)}", flush=True)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个安全领域专家。判断以下检索到的文档片段是否与用户问题相关。
只需回答 yes 或 no。"""),
        ("user", "问题：{question}\n\n检索到的文档：\n{documents}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "documents": "\n---\n".join(state["documents"][:3])  # 取前3个评估
    })
    score = "yes" if "yes" in response.content.lower() else "no"
    print(f"[grade] 判定: {score}", flush=True)
    retry = state.get("retry_count", 0)
    if score == "no":
        retry += 1
    return {"relevance_score": score, "retry_count": retry}


def rewrite_query(state: RAGState) -> dict:
    """查询改写节点：当检索不相关时，改写查询"""
    print(f"[rewrite] 改写查询，当前 retry_count={state.get('retry_count',0)}", flush=True)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个查询改写专家。原始问题在知识库中没有找到相关内容。
请基于原始问题，从不同角度重新表述查询，使其更容易检索到相关信息。
只输出改写后的查询，不要其他内容。"""),
        ("user", "原始问题：{question}\n上次查询：{last_query}")
    ])
    chain = prompt | llm
    last_query = state.get("rewritten_question") or state["question"]
    response = chain.invoke({
        "question": state["question"],
        "last_query": last_query
    })
    new_query = response.content.strip()
    print(f"[rewrite] 改写结果: {new_query}", flush=True)
    return {"rewritten_question": response.content.strip()}


def generate(state: RAGState) -> dict:
    """生成节点：基于检索到的文档生成回答"""
    print(f"[generate] 开始生成回答，文档数={len(state['documents'])}", flush=True)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个网络安全知识助手。基于以下参考资料回答用户问题。
规则：
1. 只基于提供的参考资料回答，如果资料中没有答案，明确说"知识库中未找到相关信息"
2. 回答要准确、专业，适合安全从业者阅读
3. 如果可能，指出信息来源的具体章节

参考资料：
{context}"""),
        ("user", "{question}")
    ])
    chain = prompt | llm
    context = "\n\n---\n\n".join(state["documents"])
    response = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    print(f"[generate] 回答全文: {response.content}...", flush=True)
    return {"answer": response.content}


# ============ 路由函数 ============
# def should_continue(state: RAGState) -> str:
#     """决定下一步走向"""
#     retry = state.get("retry_count", 0)
#     if state["relevance_score"] == "yes":
#         return "generate"
#     elif retry < 2:  # 最多重试2次
#         return "rewrite"
#     else:
#         return "generate"  # 超过重试次数，强制生成
def should_continue(state: RAGState) -> str:
    retry = state.get("retry_count", 0)
    decision = ""
    if state["relevance_score"] == "yes":
        decision = "generate"
    elif retry < 2:
        decision = "rewrite"
    else:
        decision = "generate"  # 超过重试次数，强制生成
    print(f"[route] relevance_score={state['relevance_score']}, retry_count={retry}, 决策: {decision}", flush=True)
    return decision

# ============ 构建 LangGraph ============
def build_rag_graph():
    """构建 RAG 工作流图"""
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    # 定义边
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges("grade_relevance", should_continue, {
        "generate": "generate",
        "rewrite": "rewrite_query",
    })
    graph.add_edge("rewrite_query", "retrieve")  # 改写后重新检索
    graph.add_edge("generate", END)

    return graph.compile()


# 编译一次，全局复用
rag_app = build_rag_graph()
# out = rag_app.invoke({
#         "question": "什么是零日漏洞？",
#         "rewritten_question": "",
#         "documents": [],
#         "answer": "",
#         "relevance_score": "",
#         "retry_count": 0,
#     })
# print(out)
