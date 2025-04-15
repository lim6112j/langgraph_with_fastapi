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
                    'description': 'The name of the file to create (can include path)',
                },
                'content': {
                    'type': 'string',
                    'description': 'The content to write to the file',
                },
                'mode': {
                    'type': 'string',
                    'description': 'File creation mode (text/binary)',
                    'enum': ['text', 'binary'],
                    'default': 'text'
                }
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
                    'description': 'The name of the file to read (can include path)',
                },
                'mode': {
                    'type': 'string',
                    'description': 'File reading mode (text/binary)',
                    'enum': ['text', 'binary'],
                    'default': 'text'
                }
            },
            'required': ['filename'],
        },
    },
}

tool_delete_file = {
    'type': 'function',
    'function': {
        'name': 'delete_file',
        'description': 'Delete a file or directory',
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The path of the file or directory to delete',
                },
                'recursive': {
                    'type': 'boolean',
                    'description': 'Recursively delete directory contents',
                    'default': False
                }
            },
            'required': ['path'],
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
                    'description': 'The name of the file to edit (can include path)',
                },
                'content': {
                    'type': 'string',
                    'description': 'The content to write to the file',
                },
                'mode': {
                    'type': 'string',
                    'description': 'File editing mode (append/overwrite)',
                    'enum': ['append', 'overwrite'],
                    'default': 'overwrite'
                }
            },
            'required': ['filename', 'content'],
        },
    },
}

tool_list_files = {
    'type': 'function',
    'function': {
        'name': 'list_files',
        'description': 'List files and directories in a given path',
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The directory path to list files from',
                    'default': '.'
                },
                'recursive': {
                    'type': 'boolean',
                    'description': 'List files recursively',
                    'default': False
                },
                'pattern': {
                    'type': 'string',
                    'description': 'File name pattern to filter results'
                }
            },
            'required': [],
        },
    },
}

tool_create_directory = {
    'type': 'function',
    'function': {
        'name': 'create_directory',
        'description': 'Create a new directory',
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The path of the directory to create',
                },
                'parents': {
                    'type': 'boolean',
                    'description': 'Create parent directories if they do not exist',
                    'default': False
                }
            },
            'required': ['path'],
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
tools = [
    tool_create_file, 
    tool_read_file, 
    tool_delete_file, 
    tool_edit_file,
    tool_list_files,
    tool_create_directory,
    tool_add_tasks_to_db, 
    tool_search_google
]
