"""LangChain v1 快速入门示例。

对应官方文档：https://langchain-doc.cn/v1/python/langchain/quickstart.html

演示如何用 LangChain v1 的 ``create_agent`` 构建一个功能完整的 AI 代理，
完整覆盖生产级代理的五个关键概念：

1. 系统提示（system_prompt）：定义代理角色与行为，保持具体且可操作。
2. 工具（tools）：让模型通过调用函数与外部系统交互；其中 ``get_user_location``
   展示了如何借助 ``ToolRuntime`` 注入运行时上下文。
3. 模型配置（init_chat_model）：选择语言模型并设置参数（temperature 等）。
4. 结构化输出（response_format）：让代理响应符合固定 schema（dataclass 或
   Pydantic 模型均可），便于下游程序稳定解析。
5. 对话记忆（checkpointer）：跨多次交互保持状态，实现类似聊天的连续对话。

运行前准备：
- 安装依赖：``pip install langchain langgraph langchain-anthropic``
  （若改用 OpenAI 模型，则安装 ``langchain-openai``）
- 在项目 ``.env`` 中设置对应的服务密钥，例如：
  ``ANTHROPIC_API_KEY=sk-ant-xxxx``
  也可通过 ``MODEL`` 环境变量切换模型，如 ``MODEL=openai:gpt-4o-mini``。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver

ENV_FILE = Path(__file__).resolve().parent / ".env"

# ──────────────────────────────────────────────────────────────
# 1. 系统提示：定义代理的角色与行为，保持具体且可操作。
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：
- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""


def load_project_env() -> None:
    """加载项目 .env 中尚未设置的变量，不依赖额外的 dotenv 包。"""
    if not ENV_FILE.is_file():
        return

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        # 兼容 `export KEY=value` 写法，否则会把 "export KEY" 整体当作变量名。
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


# ──────────────────────────────────────────────────────────────
# 2. 运行时上下文模式：工具可通过 ToolRuntime 注入获取。
# ──────────────────────────────────────────────────────────────
@dataclass
class Context:
    """自定义运行时上下文模式。"""

    user_id: str


# ──────────────────────────────────────────────────────────────
# 3. 工具：工具的名称、描述、参数名都会成为模型提示的一部分，
#    因此需要良好的文档说明。@tool 装饰器会自动添加元数据，
#    并通过 ToolRuntime 参数启用运行时注入。
# ──────────────────────────────────────────────────────────────
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city} 总是阳光明媚！"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# ──────────────────────────────────────────────────────────────
# 4. 结构化响应格式：让代理响应符合固定 schema。
#    这里使用 dataclass，也支持 Pydantic 模型。
# ──────────────────────────────────────────────────────────────
@dataclass
class ResponseFormat:
    """代理的响应模式。"""

    punny_response: str  # 带双关语的回应（始终必需）
    weather_conditions: str | None = None  # 天气的任何有趣信息（如果有）


def build_agent():
    """配置模型、记忆并组装成代理。"""
    # 模型可通过环境变量 MODEL 覆盖。默认使用 deepseek-flash（支持非思考模式）。
    # 注意：deepseek-v4-pro 仅思考模式，deepseek-flash 默认也是思考模式；
    # 思考模式不支持强制 tool_choice，会报 'Thinking mode does not support
    # this tool_choice'，因此下面通过 extra_body 关闭思考以启用结构化输出。
    model_name = os.environ.get("MODEL", "deepseek-flash")

    # 选用 DeepSeek 系列模型时，提前校验 API Key 是否已配置，
    # 避免在 init_chat_model 内部抛出晦涩的英文校验错误。
    if model_name.lower().startswith("deepseek") and not os.environ.get(
        "DEEPSEEK_API_KEY"
    ):
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY。请在项目根目录的 .env 文件中配置：\n"
            "    DEEPSEEK_API_KEY=sk-你的真实key\n"
            "（可前往 https://platform.deepseek.com 申请；配置后重新运行即可）"
        )

    model = init_chat_model(
        model=model_name,
        temperature=0.5,
        # DeepSeek 默认开启思考模式(thinking)，而思考模式不支持强制 tool_choice，
        # 会导致 ToolStrategy 结构化输出报 'Thinking mode does not support this
        # tool_choice'。通过 extra_body 关闭思考，即可启用 function calling。
        extra_body={"thinking": {"type": "disabled"}},
    )

    # 对话记忆：内存型检查点；生产环境请换成持久化的检查点（如数据库）。
    checkpointer = InMemorySaver()

    return create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_user_location, get_weather_for_location],
        context_schema=Context,
        # DeepSeek 不支持 json_schema 类型的 response_format（即默认 AutoStrategy 会走的
        # ProviderStrategy 路径，会触发 400 'This response_format type is unavailable now'）。
        # 改用 ToolStrategy（函数调用 / function calling）实现结构化输出：DeepSeek 支持
        # function calling，且这种方式不向请求里塞 response_format 字段，可正常工作。
        response_format=ToolStrategy(schema=ResponseFormat),
        checkpointer=checkpointer,
    )


def main() -> None:
    """创建代理并运行两轮对话，演示记忆与结构化输出。"""
    load_project_env()

    agent = build_agent()

    # `thread_id` 是给定对话的唯一标识符，用于隔离不同会话的记忆。
    config = {"configurable": {"thread_id": "1"}}

    # 第一轮：代理会先调用 get_user_location 获取位置，再调用
    # get_weather_for_location 查询天气，最后给出带双关语的结构化响应。
    print("\n[第 1 轮] 用户：外面的天气怎么样？")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
        config=config,
        context=Context(user_id="1"),
    )
    print("代理结构化响应：", response["structured_response"])

    # 第二轮：使用相同的 thread_id 继续对话，代理能记住上文。
    print("\n[第 2 轮] 用户：谢谢！")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "谢谢！"}]},
        config=config,
        context=Context(user_id="1"),
    )
    print("代理结构化响应：", response["structured_response"])


if __name__ == "__main__":
    main()
