# 大概总结一下
# （1）把任务给规划器，return步骤列表
# （2）遍历步骤列表，把步骤和context给执行器，
# （3）执行结果追加到context和results中，context用于记录步骤结果
# （4）整合results
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 规划器
plan_prompt = PromptTemplate(
    input_variables=["task"],
    template="""你是一个任务规划专家。请将以下任务分解为具体的执行步骤：

任务：{task}

请输出 JSON 格式的步骤列表：
{{
    "steps": [
        "步骤1：...",
        "步骤2：...",
        "步骤3：..."
    ]
}}"""
)

planner = LLMChain(llm=llm, prompt=plan_prompt)

# 执行器
executor_prompt = PromptTemplate(
    input_variables=["step", "context"],
    template="""执行以下步骤：{step}

上下文信息：{context}

请执行该步骤并返回结果。"""
)

executor = LLMChain(llm=llm, prompt=executor_prompt)


# 主流程
def plan_and_execute(task):
    # 1. 规划
    plan = planner.run(task)
    steps = json.loads(plan)["steps"]

    # 2. 执行
    context = ""
    results = []
    for step in steps:
        result = executor.run(step=step, context=context)
        results.append(result)
        context += f"\n步骤结果：{result}"

    # 3. 整合
    return synthesize_results(results)
