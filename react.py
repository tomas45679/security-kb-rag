# 小小总结一下
# （1）先注册工具，弄一个工具列表
# （2）初始化模型，给模型系统提示词，这个要写全一点，然后设置轮次开始循环
# （3）把任务和对话历史给大模型，得到结果追加到对话历史中
# （4）正则判断是否调用工具，没有则直接输出结果
# （5）如果需要调用工具，则判断该工具是否在工具列表中，调用后返回结果赋值给question

import json
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
import sqlite3
import re
import datetime

api_url = os.getenv('GUIJI_API_URL')
token = os.getenv('GUIJI_API_KEY')
serper_key = os.getenv('SUPER_SER_KEY')
client = OpenAI(
    api_key=token,  # 从 https://cloud.siliconflow.cn/i/nRDJFg4z 获取
    base_url="https://api.siliconflow.cn/v1"
)


def fetch_real_time_info(query):
    """
    通过Serper API获取实时搜索信息

    使用Serper的Google搜索API查询指定问题，并返回最相关结果的摘要片段。

    参数:
        query (str): 要搜索的查询字符串，表示用户提出的问题

    返回:
        str: 搜索结果摘要片段。若无结果则返回提示信息
    """
    # API参数配置
    params = {
        'api_key': serper_key,  # 使用您自己的API密钥
        'q': query,  # 查询参数，表示要搜索的问题。
        'num': 1  # 返回结果的数量设为1，API将返回一个相关的搜索结果。
    }
    # 发起API请求
    api_result = requests.get('https://google.serper.dev/search', params)

    # 解析JSON响应数据
    search_data = api_result.json()

    # 提取并返回查询到的信息
    if search_data["organic"]:
        return search_data["organic"][0]["snippet"]
    else:
        return "没有找到相关结果。"


def calculate(operation: str) -> float:
    """
    计算字符串数学表达式的值

    使用内置eval()函数执行传入的数学表达式字符串，并返回浮点数结果。
    注意：eval()会执行任何传入的Python代码，存在严重安全风险，
    请确保仅传入可信来源的数学表达式。

    Args:
        operation: 包含数学表达式的字符串，例如 "2 * 3 + 4"

    Returns:
        表达式计算结果的浮点数值

    Raises:
        SyntaxError: 当传入字符串不是合法表达式时
        TypeError: 当表达式结果无法转换为浮点数时
        ZeroDivisionError: 当表达式中包含除以零操作时
    """
    return eval(operation)


def get_current_time():
    """
    获取当前系统时间并格式化为字符串

    返回:
        str: 格式为'YYYY-MM-DD HH:MM:SS'的当前时间字符串
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 可用操作映射字典
# 该字典定义了系统支持的操作名称与对应处理函数的映射关系
# 键：操作名称字符串，用于标识具体操作
# 值：实际执行操作的函数对象
available_actions = {
    # 实时信息获取操作：调用fetch_real_time_info函数处理
    "fetch_real_time_info": fetch_real_time_info,

    # 计算操作：调用calculate函数处理
    "calculate": calculate,
    # 获取当前时间操作：调用get_current_time函数处理
    "get_current_time": get_current_time,
}


class ChatBot:
    """
    聊天机器人实现类，用于处理用户消息并获取AI回复。
    属性:
        system (str): 系统级初始化指令，用于设定AI行为
        messages (list): 存储所有对话消息的列表，包含角色和内容
    """

    def __init__(self, system=""):
        """
        初始化聊天机器人实例

        参数:
            system: 系统级初始化指令，用于设定AI行为（默认空字符串）
        """
        self.system = system
        self.messages = []
        # 如果存在系统指令，添加到消息列表开头
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        """
        使实例可调用，处理用户消息并返回AI回复

        参数:
            message: 用户输入的文本消息

        返回:
            str: AI生成的回复内容
        """
        # 将用户消息添加到对话历史
        self.messages.append({"role": "user", "content": message})
        # 调用模型执行并获取回复
        result = self.execute()
        # 将AI回复添加到对话历史
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        """
        调用AI模型API执行对话生成

        返回:
            str: 模型生成的文本回复内容
        """
        # 创建API请求获取模型回复
        completion = client.chat.completions.create(model="Qwen/Qwen3-32B", messages=self.messages, temperature=0.5)
        # 提取模型返回的首选回复内容
        return completion.choices[0].message.content


system_prompt = """
        You run in a loop of Thought, Action, Observation, Answer.
        At the end of the loop you output an Answer
        Use Thought to describe your thoughts about the question you have been asked.
        Use Action to run one of the actions available to you.
        Observation will be the result of running those actions.
        Answer will be the result of analysing the Observation.
        Function calls MUST be used when calculating the current time.
        Your available actions are:
        get_current_time:
        e.g. get_current_time: ''
        Function: Retrieve the current time
        Operation Principle: Get current time for subsequent logical judgments and operations

        calculate:
        e.g. calculate: 4 * 7 / 3
        Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary
        fetch_real_time_info:
        e.g. fetch_real_time_info: 阿里
        Returns a real info from searching SerperAPI
        Always look things up on fetch_real_time_info if you have the opportunity to do so.
        Example session:
        Question: What is the capital of China?
        Thought: I should look up on SerperAPI
        Action: fetch_real_time_info: What is the capital of China?
        PAUSE 
        You will be called again with this:
        Observation: China is a country. The capital is Beijing.
        Thought: I think I have found the answer
        Action: Beijing.
        You should then call the appropriate action and determine the answer from the result
        You then output:
        Answer: The capital of China is Beijing
        Example session
        Question: What is the mass of Earth times 2?
        Thought: I need to find the mass of Earth on fetch_real_time_info
        Action: fetch_real_time_info : mass of earth
        PAUSE
        You will be called again with this: 
        Observation: mass of earth is 1,1944×10e25
        Thought: I need to multiply this by 2
        Action: calculate: 5.972e24 * 2
        PAUSE
        You will be called again with this: 
        Observation: 1,1944×10e25
        If you have the answer, output it as the Answer.
        Answer: The mass of Earth times 2 is 1,1944×10e25.
        Example session
        Question: Need to obtain current time for year calculation
        Thought: Need current year
        Action: get_current_time
        Observation: Output 2025-12-01 09:01:01
        Answer: 2025-12-01 09:01:01
        Now it's your turn:
            """.strip()


def AgentExecutor(question, max_turns=5):
    """
    执行多轮对话代理的核心逻辑

    通过有限轮次的对话交互处理用户问题，当模型输出包含工具调用指令时，
    自动执行相应工具并观察结果，否则返回完整对话记录

    Args:
        question (str): 用户输入的初始问题
        max_turns (int, optional): 最大对话轮次数，默认5轮

    Returns:
        list: 当未触发工具调用时返回完整对话记录
        str: 当触发工具调用时返回最后观察结果(循环中)
    """
    i = 0
    # 初始化对话机器人并加载系统预设
    bot = ChatBot(system_prompt)
    # 通过 next_prompt 标识每一个子任务的阶段性输入
    next_prompt = question
    # 解析AI返回结果中的动作指令
    action_re = re.compile(r'^Action: (\w+)(?::\s*(.*))?$', re.MULTILINE)

    # 限制最大交互轮次防止无限循环
    while i < max_turns:
        i += 1
        # 调用ChatBot模型获取当前轮次响应
        result = bot(next_prompt)
        print(f"result:{result}")

        actions = action_re.findall(result)

        # 检测到工具调用指令时执行
        if actions:
            # 解析工具名称和调用参数
            action, action_input = actions[0]  # 取第一个动作指令

            # 验证工具是否在允许列表中
            if action not in available_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))

            print(f"running: {action} {action_input}")
            if action_input != "":
                # 执行工具并获取返回结果
                observation = available_actions[action](action_input)
            else:
                # 执行工具并获取返回结果
                observation = available_actions[action]()
            print(f"Observation: {observation}")

            # 将工具结果作为下一轮输入
            next_prompt = "Observation: {}".format(observation)
        else:
            # 未检测到工具指令则终止流程并返回对话记录
            return bot.messages


if __name__ == "__main__":
    # react_test_call()
    # res = fetch_real_time_info("长沙岳麓山")
    # print(res)
    # print(get_current_time())
    AgentExecutor("阿里成立多少年了？")
    # test_split()
    # test_split2()
