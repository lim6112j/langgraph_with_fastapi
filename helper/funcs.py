from langchain_core.messages import SystemMessage
from messages.messages import template, template_api_automation_agent, template_menu_agent, template_crawl_agent, template_dashboard_agent


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


def get_messages_menu_info(messages):
    return [SystemMessage(content=template_menu_agent)] + messages


def get_messages_crawl_info(messages):
    return [SystemMessage(content=template_crawl_agent)] + messages


def get_messages_dashboard_info(messages):
    return [SystemMessage(content=template_dashboard_agent)] + messages


def get_messages_api_automation_info(messages):
    return [SystemMessage(content=template_api_automation_agent)] + messages
