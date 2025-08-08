from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage

def get_prompt_from_msgs(messages: List[AnyMessage]) -> str | list[str | Dict[str, str]]:
    """
    Get the prompt from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        prompt_context = messages[-1].content
    else:
        prompt_context = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt_context += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt_context += f"Assistant: {message.content}\n"
    return prompt_context
