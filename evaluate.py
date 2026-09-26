"""
RAG 效果评测脚本
评测维度：忠实度（Faithfulness）、相关性（Relevance）、回答率
"""
import json
import time
from rag import rag_app
from config import *


# ============ 测试集：手动构建安全领域的 QA 对 ============
TEST_CASES = [
    {
        "question": "SQL注入攻击有哪些常见类型？",
        "expected_keywords": ["联合查询", "盲注", "报错注入", "时间盲注"],
        "category": "基础概念"
    },
    {
        "question": "如何进行渗透测试的信息收集阶段？",
        "expected_keywords": ["子域名", "端口扫描", "whois", "DNS"],
        "category": "渗透测试"
    },
    {
        "question": "OWASP Top 10 中排名第一的漏洞是什么？",
        "expected_keywords": ["注入", "Broken Access", "失效"],
        "category": "安全标准"
    },
    {
        "question": "XSS攻击的三种类型分别是什么？",
        "expected_keywords": ["反射型", "存储型", "DOM型"],
        "category": "Web安全"
    },
    {
        "question": "什么是零日漏洞？",
        "expected_keywords": ["0day", "未公开", "补丁"],
        "category": "基础概念"
    },
]


def evaluate_single(test_case: dict) -> dict:
    """评测单个问题"""
    start = time.time()
    result = rag_app.invoke({
        "question": test_case["question"],
        "rewritten_question": "",
        "documents": [],
        "answer": "",
        "relevance_score": "",
        "retry_count": 0,
    })
    elapsed = time.time() - start

    # 关键词命中率
    answer = result["answer"]
    hit_keywords = [kw for kw in test_case["expected_keywords"] if kw in answer]
    keyword_recall = len(hit_keywords) / len(test_case["expected_keywords"])

    # 是否拒答（知识库中没有时）
    is_refused = "未找到" in answer or "无法回答" in answer

    return {
        "question": test_case["question"],
        "category": test_case["category"],
        "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
        "keyword_recall": round(keyword_recall, 2),
        "hit_keywords": hit_keywords,
        "is_refused": is_refused,
        "retrieved_count": len(result["documents"]),
        "time_seconds": round(elapsed, 2),
    }


def run_evaluation():
    """运行完整评测"""
    print("=" * 60)
    print("🔍 安全知识库 RAG 效果评测")
    print("=" * 60)

    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {tc['question']}")
        result = evaluate_single(tc)
        results.append(result)
        print(f"  ⏱️  耗时: {result['time_seconds']}s")
        print(f"  📊 关键词召回: {result['keyword_recall']} ({result['hit_keywords']})")
        print(f"  📝 回答预览: {result['answer_preview'][:80]}...")

    # 汇总统计
    avg_recall = sum(r["keyword_recall"] for r in results) / len(results)
    avg_time = sum(r["time_seconds"] for r in results) / len(results)
    avg_docs = sum(r["retrieved_count"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("📊 评测汇总")
    print(f"  平均关键词召回率: {avg_recall:.2%}")
    print(f"  平均响应时间: {avg_time:.2f}s")
    print(f"  平均检索文档数: {avg_docs:.1f}")
    print("=" * 60)

    # 保存详细结果
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("💾 详细结果已保存到 eval_results.json")

    return results


if __name__ == "__main__":
    run_evaluation()