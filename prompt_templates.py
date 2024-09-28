memory_prompt_template = """<|im_start|>system
You are an AI chatbot having a conversation with a human. Answer his questions.<|im_end|>
<|im_start|>user
{history}
{human_input}<|im_end|>
<|im_start|>assistant
"""
