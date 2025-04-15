from openhands import LLM
from openhands.llm import MyLocalModel  # 假设你有一个 MyLocalModel 类来加载本地模型

# 配置本地模型
llm = MyLocalModel(model_path="path/to/your/local/model")

# 通过 CodeAct 启动代理
agent = LLM(agent_type="codeact", llm=llm)