class LLMAgent:
    def __init__(self, llm_agent_type):
        # Call the output code from LLM agents file
        if llm_agent_type == "AzureGPT4Agent":
            self.llm_agent = AzureGPT4Agent()
        elif llm_agent_type == "GoogleGeminiAgent":
            self.llm_agent = GoogleGeminiAgent()
        elif llm_agent_type == "Qwen/QwQ-32B-Preview":
            self.llm_agent = QwQModel()
        elif llm_agent_type == "meta-llama/Meta-Llama-3.1-70B-Instruct":
            self.llm_agent = LlamaModel()
        elif llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
            self.llm_agent = QwenModel()
    