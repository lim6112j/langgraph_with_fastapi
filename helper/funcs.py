from langchain_core.messages import SystemMessage
from messages.messages import template, template_menu_agent
def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages
def get_messages_menu_info(messages):
    return [SystemMessage(content=template_menu_agent)] + messages
