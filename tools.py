tool_create_file = {
    'type': 'function',
    'function': {
        'name': 'create_file',
        'description': 'Create a new file with given content',
        'parameters': {
            'type': 'object',
            'properties': {
                'filename': {
                    'type': 'string',
                    'description': 'The name of the file to create',
                },
                'content': {
                    'type': 'string',
                    'description': 'The content to write the file',
                },
            },
            'required': ['filename', 'content']
        },
    },
}

tool_read_file = {
    'type': 'function',
    'function': {
        'name': 'read_file',
        'description': 'Read the content of a file',
        'parameters': {
            'type': 'object',
            'properties': {
                'filename': {
                    'type': 'string',
                    'description': 'The name of the file to read',
                    },
                },
            'required': ['filename'],
        },
    },
}

tool_delete_file = {
    'type': 'function',
    'function': {
        'name': 'delete_file',
        'description': 'Delete a file',
        'parameters': {
            'type': 'object',
            'properties': {
                'filename': {
                    'type': 'string',
                    'description': 'The name of the file to delete',
                },
            },
            'required': ['filename'],
        },
    },
}

tool_edit_file = {
    'type': 'function',
    'function': {
        'name': 'edit_file',
        'description': 'Edit the content of a file',
        'parameters': {
            'type': 'object',
            'properties': {
                'filename': {
                    'type': 'string',
                    'description': 'The name of the file to edit',
                },
                'content': {
                    'type': 'string',
                    'description': 'The content to write to the file',
                },
            },
            'required': ['filename', 'content'],
        },
    },
}

tool_add_tasks_to_db = {
    'type': 'function',
    'function': {
        'name': 'add_task',
        'description': 'Add a task to the tasks database',
        'parameters': {
            'type': 'object',
            'properties': {
                'task_description': {
                    'type': 'string',
                    'description': 'The description of the task to add',
                },
            },
            'required': ['task_description']
        },
    },
}
tool_search_google = {
    'type': 'function',
    'function': {
        'name': 'search_google',
        'description': 'search from google',
        'parameters': {
            'type': 'object',
            'properties': {
                'keyword': {
                    'type': 'string',
                    'description': 'keyword for searching'
                },
            },
            'required': ['keyword'],
        },
    },

}
tools = [tool_create_file, tool_read_file, tool_delete_file, tool_add_tasks_to_db, tool_search_google]
