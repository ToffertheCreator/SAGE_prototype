from prompt_templates import memory_prompt_template
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_llm(model_path=config['model_path']['large'], model_type=config['model_type'], model_config=config['model_config']):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm

def create_embeddings(embeddings_path=config['embeddings_path']):
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key='history', chat_memory=chat_history, k=5)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

def load_normal_chain(chat_history):
    system_message = "You are an AI chatbot having a conversation with a human. Answer his questions."
    return chatChain(chat_history, system_message=system_message)

    
class chatChain:
    def __init__(self, chat_history, system_message):
        self.memory = create_chat_memory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)
        self.system_message = system_message
    
    def run(self, user_input):
        # Prepare the formatted prompt
        formatted_prompt = self.llm_chain.prompt.format(
            system_message=self.system_message,
            history=self.memory.chat_memory.messages,
            human_input=user_input
        )
        
        # Generate the response
        response = self.llm_chain.llm(
            formatted_prompt,
            stop=['<|im_start|>user']  # Stop generation when these tokens are encountered
        )
        
        # Ensure response is not None
        if response:
            # Add the conversation to memory
            self.memory.save_context({"input": user_input}, {"output": response})
        
        return response
