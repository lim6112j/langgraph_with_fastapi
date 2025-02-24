from langchain_core.messages import SystemMessage
from messages.messages import template
def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages
