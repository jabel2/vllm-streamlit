############################  TOOL CALLING AGENT ######################################################################################################
######################## https://api.python.langchain.com/en/latest/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html ####
from langchain import hub
from langchain.agents import AgentExecutor, load_tools, tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (ReActJsonSingleInputOutputParser,)
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.tools import Tool
from langchain.tools.base import ToolException

from vllm_interface import llm

#chat_model = ChatHuggingFace(llm=llm)
@tool
def word_length(word: str) -> int:
    """Returns a counter word"""
    return len(word)

@tool
def favorite_animal(name: str) -> str:
    """Get the favorite animal of the person with the given name"""
    if name.lower().strip() == "eugene":
        return "cat"
    return "dog"

# setup tools
tools = [
    Tool.from_function(
        func=word_length,
        name="word length",
        description="Used to determine the length of a word.",
        handle_tool_error=True,
    ),
    Tool.from_function(
        func=favorite_animal,
        name="favorite_animal",
        description="favorite_animal",
        handle_tool_error=True,
    ),
]

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
chat_model_with_stop = llm.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent_executor.invoke(
    {
        "input": "What is the length of the word 'testredsat'?"
    }
)