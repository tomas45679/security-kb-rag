# 我总结了一下大概步骤
# （1）输入任务和memory给执行器
# （2）把任务和执行结果给评估器，通过直接return结果
# （3）把任务、执行结果、评估结果给反思器，
# （4）把反思结果追加到memory中，进行下一轮循环

class ReflexionAgent:
    def __init__(self):
        self.memory = []  # 存储反思记忆
        self.max_iterations = 3

    def run(self, task):
        for iteration in range(self.max_iterations):
            # 1. 执行任务
            result = self.actor.run(task, memory=self.memory)

            # 2. 评估结果
            evaluation = self.evaluator.evaluate(task, result)

            # 3. 判断是否成功
            if evaluation.is_success:
                return result

            # 4. 反思失败原因
            reflection = self.reflect(task, result, evaluation)
            self.memory.append(reflection)

            print(f"第 {iteration + 1} 轮失败，反思：{reflection}")

        return result  # 返回最后一次结果

    def reflect(self, task, result, evaluation):
        """生成反思内容"""
        prompt = f"""任务：{task}
执行结果：{result}
评估反馈：{evaluation.feedback}

请分析失败原因，并提出改进建议："""

        return self.llm.generate(prompt)
