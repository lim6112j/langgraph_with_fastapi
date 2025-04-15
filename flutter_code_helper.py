import os
import shutil
import re
import subprocess
import logging
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import getpass
import json
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flutter_code_helper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
# OpenAI API 키 설정
if not os.getenv("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = getpass.getpass(
            "Enter Api key for OPENAI")
        logger.info("OpenAI API key successfully set")
    except Exception as e:
        logger.error(f"Failed to set OpenAI API key: {e}")
        raise

# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0.2, max_tokens=4096)

# 대화 메모리 초기화
memory = ConversationBufferMemory()

# Flutter 작업을 위한 프롬프트 템플릿 (BLoC 기반)
flutter_prompt = PromptTemplate(
    input_variables=["request", "history", "filename", "project_context"],
    template="""
    You are an expert Flutter developer familiar with the BLoC pattern and clean architecture principles. 
    
    Enhanced Flutter Code Generation Context:
    - Strict adherence to clean architecture principles (presentation, domain, data layers)
    - Follow latest Flutter best practices (Flutter 3.10+)
    - Consider performance and scalability in your implementation
    - Implement robust error handling with try/catch blocks
    - Use modern Dart language features (null safety, extension methods, etc.)
    - Ensure code is testable with clear separation of concerns
    
    Project Context: {project_context}
    Conversation history: {history}
    Target Filename: {filename}
    User Request: {request}
    
    Generate production-ready code addressing:
    1. Code quality - Follow consistent naming conventions and formatting
    2. Performance considerations - Avoid expensive operations in build methods
    3. Maintainability - Use clear comments and documentation
    4. Scalability - Design for future feature additions
    5. Error resilience - Handle edge cases and network failures
    
    Use the `flutter_bloc` package for state management. Include necessary imports, widgets, BLoC setup (Bloc, Event, State), and appropriate styling. 
    
    If generating code, provide it in a markdown code block (```dart) for the file named '{filename}'. The code must be a complete, self-contained widget or class suitable for a separate file, not embedded in `main.dart`. Ensure proper Dart syntax and avoid including `main()` or `runApp` unless the filename is `main.dart`.
    
    Output the result (code in ```dart block if applicable, otherwise plain text).
    """
)

# 챗봇 요청 분석을 위한 프롬프트 템플릿
chatbot_prompt = PromptTemplate(
    input_variables=["request", "history"],
    template="""
    You are a chatbot managing a Flutter project workflow. Analyze the user's request and conversation history to determine the action to take and the target filename. The possible actions are:
    - "create_project": Create a new Flutter project with the specified requirements.
    - "process_existing": Analyze or modify code in a specific file.
    - "run_app": Run the existing Flutter app.
    - "other": Provide general information or clarification.

    If the request implies creating or modifying a specific screen (e.g., "login screen"), suggest a filename following clean architecture principles:
    - UI/Screens: `lib/presentation/screens/<screen_name>_screen.dart`
    - Widgets: `lib/presentation/widgets/<widget_name>.dart`
    - BLoC: `lib/presentation/bloc/<feature>/<feature>_bloc.dart`
    - Models: `lib/domain/models/<model_name>.dart`
    - Repositories: `lib/domain/repositories/<repository_name>.dart`
    - Data sources: `lib/data/datasources/<source_name>.dart`

    For example, a login screen request should create:
    - `lib/presentation/screens/login_screen.dart` for the UI
    - `lib/presentation/bloc/auth/auth_bloc.dart` for the BLoC
    - `lib/domain/repositories/auth_repository.dart` for the repository

    If no specific file is implied or the request is ambiguous, default to `lib/main.dart`.

    Conversation history: {history}

    User's request: {request}

    Output a JSON object with:
    - "action": One of ["create_project", "process_existing", "run_app", "other"]
    - "processed_request": The request to pass to the selected action
    - "filename": The target filename (e.g., "lib/presentation/screens/login_screen.dart" or "lib/main.dart")
    - "feature_type": The type of feature being requested (e.g., "login", "dashboard", "settings", etc.)
    """
)

# LangChain 체인 생성
flutter_chain = LLMChain(llm=llm, prompt=flutter_prompt)
chatbot_chain = LLMChain(llm=llm, prompt=chatbot_prompt)

# Code validation utilities


def lint_code(code_snippet):
    """
    Perform basic linting on Dart code
    """
    # Check for common issues
    issues = []

    # Check for missing semicolons
    if re.search(r'}\s*$', code_snippet, re.MULTILINE):
        issues.append("Missing semicolons")

    # Check for proper null safety
    if "?" not in code_snippet and "late " not in code_snippet and "required" not in code_snippet:
        issues.append("Null safety features not properly utilized")

    # Check for error handling
    if "try" not in code_snippet and "catch" not in code_snippet:
        issues.append("No error handling found")

    # Check for proper BLoC pattern usage if it's a BLoC file
    if "bloc" in code_snippet.lower() and "emit(" not in code_snippet:
        issues.append("BLoC not properly emitting states")

    return len(issues) == 0


def check_best_practices(code_snippet):
    """
    Check if code follows Flutter best practices
    """
    best_practices = [
        # Proper widget structure
        "extends StatelessWidget" in code_snippet or "extends StatefulWidget" in code_snippet,

        # Proper imports
        "import 'package:flutter/" in code_snippet,

        # Proper constructor
        "const " in code_snippet and "({" in code_snippet,

        # Key usage
        "Key?" in code_snippet,

        # Documentation
        "///" in code_snippet or "/*" in code_snippet
    ]

    # At least 3 best practices should be followed
    return sum(best_practices) >= 3


def validate_generated_code(code_snippet):
    """
    Validate generated code against multiple criteria
    """
    validation_checks = [
        lint_code(code_snippet),           # Static analysis
        check_best_practices(code_snippet),  # Design pattern compliance
    ]

    return all(validation_checks)


class CodeGenerationContext:
    """
    Provides context for code generation based on feature type
    """

    def __init__(self):
        self.design_patterns = {
            "login": ["BLoC", "Repository Pattern"],
            "dashboard": ["BLoC", "Repository Pattern"],
            "settings": ["BLoC", "Repository Pattern"],
            "list": ["BLoC", "Repository Pattern"],
            "detail": ["BLoC", "Repository Pattern"],
            "form": ["BLoC", "Form Validation"],
        }

    def get_recommended_architecture(self, feature_type):
        """
        Get recommended architecture based on feature type
        """
        return self.design_patterns.get(feature_type, ["BLoC"])

    def get_project_context(self, feature_type, filename):
        """
        Generate project context based on feature type and filename
        """
        architecture = self.get_recommended_architecture(feature_type)

        context = f"Feature type: {feature_type}\n"
        context += f"Recommended architecture: {', '.join(architecture)}\n"

        if "screen" in filename:
            context += "This is a UI component that should focus on presentation logic only.\n"
            context += "Business logic should be delegated to the BLoC.\n"
        elif "bloc" in filename:
            context += "This is a BLoC component that should handle business logic.\n"
            context += "It should process events and emit states.\n"
        elif "repository" in filename:
            context += "This is a repository that should abstract data sources.\n"
            context += "It should provide a clean API for the domain layer.\n"
        elif "model" in filename:
            context += "This is a model that should represent domain entities.\n"
            context += "It should include proper serialization/deserialization.\n"

        return context


# Initialize code generation context
code_context = CodeGenerationContext()

# LangGraph 상태 정의


class GraphState(Dict[str, Any]):
    directory: str
    request: str
    processed_request: str
    action: str
    filename: str
    project_dir: str
    status: str
    code: str
    feature_type: str
    project_context: str

# pubspec.yaml에 의존성 추가 함수


def add_dependency_to_pubspec(pubspec_path: str, dependency: str, version: str) -> bool:
    """
    Add dependency to pubspec.yaml with enhanced error handling and logging

    Args:
        pubspec_path (str): Path to pubspec.yaml
        dependency (str): Name of the dependency
        version (str): Version of the dependency

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(pubspec_path):
            logger.error(f"Pubspec file not found: {pubspec_path}")
            return False

        with open(pubspec_path, "r") as f:
            pubspec = yaml.safe_load(f) or {}

        pubspec.setdefault("dependencies", {})
        pubspec["dependencies"][dependency] = version

        with open(pubspec_path, "w") as f:
            yaml.dump(pubspec, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Added dependency {dependency}@{version} to pubspec.yaml")
        return True
    except (IOError, PermissionError) as e:
        logger.error(f"File access error in pubspec: {e}")
        return False
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating pubspec.yaml: {e}")
        return False

# LangGraph 노드 정의


def chatbot_node(state: GraphState) -> GraphState:
    """사용자 요청을 분석하는 챗봇 노드"""
    try:
        response = chatbot_chain.invoke({
            "request": state["request"],
            "history": memory.buffer
        })["text"]
        response_json = json.loads(response)
        state["action"] = response_json["action"]
        state["processed_request"] = response_json["processed_request"]
        state["filename"] = response_json["filename"]
        state["feature_type"] = response_json.get("feature_type", "general")

        # Generate project context based on feature type and filename
        state["project_context"] = code_context.get_project_context(
            state["feature_type"],
            state["filename"]
        )

        return state
    except Exception as e:
        state["status"] = f"Error in chatbot: {e}"
        state["action"] = "other"
        state["filename"] = "lib/main.dart"
        state["feature_type"] = "general"
        state["project_context"] = "General Flutter application"
        return state


def create_project_node(state: GraphState) -> GraphState:
    """
    Flutter 프로젝트 생성 노드 with enhanced error handling and validation

    Args:
        state (GraphState): Current workflow state

    Returns:
        GraphState: Updated state with project creation results
    """
    try:
        # Validate input directory
        directory = os.path.abspath(state.get("directory", ""))
        if not directory or not os.access(directory, os.W_OK):
            logger.error(f"Invalid or non-writable directory: {directory}")
            state["status"] = "Invalid project directory"
            return state

        project_dir = os.path.join(directory, "my_flutter_app")

        # Safely create directory
        try:
            os.makedirs(directory, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied creating directory: {directory}")
            state["status"] = "Permission denied"
            return state

        # Check Flutter CLI
        if not shutil.which("flutter"):
            logger.error("Flutter CLI not installed")
            state["status"] = "Flutter CLI not found"
            return state

        # Remove existing project if it exists
        if os.path.exists(project_dir):
            try:
                shutil.rmtree(project_dir)
            except PermissionError:
                logger.error(f"Cannot remove existing project: {project_dir}")
                state["status"] = "Cannot remove existing project"
                return state

        # Create Flutter project with error handling
        create_result = subprocess.run(
            ["flutter", "create", "--quiet", project_dir],
            capture_output=True,
            text=True
        )
        if create_result.returncode != 0:
            logger.error(
                f"Flutter project creation failed: {create_result.stderr}")
            state["status"] = f"Project creation error: {create_result.stderr}"
            return state

        # Add dependencies
        pubspec_path = os.path.join(project_dir, "pubspec.yaml")
        if not add_dependency_to_pubspec(pubspec_path, "flutter_bloc", "^8.1.3"):
            logger.warning("Could not add flutter_bloc dependency")

        # Pub get with error handling
        pub_result = subprocess.run(
            ["flutter", "pub", "get"],
            cwd=project_dir,
            capture_output=True,
            text=True
        )
        if pub_result.returncode != 0:
            logger.error(f"Pub get failed: {pub_result.stderr}")

        # 요청된 파일 생성
        response = flutter_chain.invoke({
            "request": state["processed_request"],
            "history": memory.buffer,
            "filename": state["filename"],
            "project_context": state["project_context"]
        })["text"]
        code_match = re.search(r"```dart\n([\s\S]*?)\n```", response)
        dart_code = code_match.group(1) if code_match else response

        # Validate and improve code if needed
        if code_match and not validate_generated_code(dart_code):
            # Try to improve the code with more specific instructions
            improved_response = flutter_chain.invoke({
                "request": f"Improve the following code to follow best practices, include proper error handling, and use null safety correctly:\n```dart\n{dart_code}\n```",
                "history": memory.buffer,
                "filename": state["filename"],
                "project_context": state["project_context"]
            })["text"]
            improved_code_match = re.search(
                r"```dart\n([\s\S]*?)\n```", improved_response)
            if improved_code_match:
                dart_code = improved_code_match.group(1)

        # 파일 경로 설정
        file_path = os.path.join(project_dir, state["filename"])
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(dart_code.strip())
        except (IOError, PermissionError) as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            state["status"] = f"File write error: {e}"
            return state

        # main.dart 기본 코드 생성 (필요 시)
        main_dart_path = os.path.join(project_dir, "lib", "main.dart")
        if state["filename"] != "lib/main.dart":
            try:
                class_name = ''.join(word.capitalize() for word in os.path.basename(
                    state["filename"]).replace(".dart", "").split('_'))
                main_code = f"""import 'package:flutter/material.dart';
import '{state["filename"].replace("lib/", "./")}';

void main() {{
  runApp(MyApp());
}}

class MyApp extends StatelessWidget {{
  @override
  Widget build(BuildContext context) {{
    return MaterialApp(
      home: {class_name}(),
    );
  }}
}}
"""
                with open(main_dart_path, "w") as f:
                    f.write(main_code.strip())
            except (IOError, PermissionError) as e:
                logger.warning(f"Failed to create main.dart: {e}")

        logger.info(f"Flutter project created successfully at {project_dir}")
        state["status"] = f"Flutter 프로젝트가 {project_dir}에 생성되었습니다. 파일: {state['filename']}"
        state["code"] = dart_code.strip()
        state["project_dir"] = project_dir
        return state

    except Exception as e:
        logger.exception("Unexpected error in project creation")
        state["status"] = f"Unexpected error: {str(e)}"
        state["code"] = None
        return state


def process_existing_project_node(state: GraphState) -> GraphState:
    """기존 프로젝트 작업 노드"""
    try:
        project_dir = state["project_dir"]
        if not project_dir or not os.path.exists(project_dir):
            raise Exception("유효한 프로젝트 디렉토리가 없습니다.")

        file_path = os.path.join(project_dir, state["filename"])
        existing_code = ""
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                existing_code = f.read()
            memory.chat_memory.add_user_message(
                f"Previous {state['filename']} code:\n```dart\n{existing_code}\n```")

        response = flutter_chain.invoke({
            "request": state["processed_request"],
            "history": memory.buffer,
            "filename": state["filename"],
            "project_context": state["project_context"]
        })["text"]
        code_match = re.search(r"```dart\n([\s\S]*?)\n```", response)
        if code_match:
            dart_code = code_match.group(1)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(dart_code.strip())
            state["status"] = f"{state['filename']}가 업데이트되었습니다: {file_path}"
            state["code"] = dart_code.strip()
        else:
            state["status"] = response
            state["code"] = None
        return state
    except Exception as e:
        state["status"] = f"Error: {e}"
        state["code"] = None
        return state


def run_app_node(state: GraphState) -> GraphState:
    """Flutter 앱 실행 노드"""
    try:
        project_dir = state["project_dir"]
        if not project_dir or not os.path.exists(project_dir):
            raise Exception("실행할 프로젝트가 없습니다.")

        process = subprocess.Popen(
            ["flutter", "run", "-d", "emulator"],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=120)
        if process.returncode == 0:
            state["status"] = f"앱이 성공적으로 실행되었습니다.\nOutput:\n{stdout}"
        else:
            state["status"] = f"앱 실행 실패:\nError:\n{stderr}"

        file_path = os.path.join(
            project_dir, state.get("filename", "lib/main.dart"))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                state["code"] = f.read().strip()
        return state
    except subprocess.TimeoutExpired:
        process.kill()
        state["status"] = "Error: 앱 실행이 30초 내에 완료되지 않았습니다."
        return state
    except Exception as e:
        state["status"] = f"Error: {e}"
        return state


def other_action_node(state: GraphState) -> GraphState:
    """기타 요청 처리 노드"""
    response = flutter_chain.invoke({
        "request": state["processed_request"],
        "history": memory.buffer,
        "filename": state["filename"]
    })["text"]
    state["status"] = response
    state["code"] = None
    return state


# LangGraph 워크플로우 정의
workflow = StateGraph(GraphState)

workflow.add_node("chatbot", chatbot_node)
workflow.add_node("create_project", create_project_node)
workflow.add_node("process_existing", process_existing_project_node)
workflow.add_node("run_app", run_app_node)
workflow.add_node("other", other_action_node)

workflow.set_entry_point("chatbot")
workflow.add_conditional_edges(
    "chatbot",
    lambda state: state["action"],
    {
        "create_project": "create_project",
        "process_existing": "process_existing",
        "run_app": "run_app",
        "other": "other"
    }
)
workflow.add_edge("create_project", END)
workflow.add_edge("process_existing", END)
workflow.add_edge("run_app", END)
workflow.add_edge("other", END)

app_graph = workflow.compile()

# Gradio 인터페이스 처리 함수


def gradio_interface(directory, request, project_dir_state):
    initial_state = GraphState(
        directory=directory,
        request=request,
        processed_request=request,
        action="",
        filename="lib/main.dart",
        project_dir=project_dir_state,
        status="",
        code="",
        feature_type="general",
        project_context="General Flutter application"
    )
    result = app_graph.invoke(initial_state)
    return result["status"], result["code"], result["project_dir"]


# Gradio 앱 실행
with gr.Blocks(title="Flutter Project Manager with Chatbot") as app:
    gr.Markdown("# Flutter 프로젝트 관리기 (BLoC 기반)")
    gr.Markdown("챗봇이 요청을 분석하여 프로젝트 생성, 코드 분석, 수정, 실행 등을 처리합니다.")

    directory_input = gr.Textbox(
        label="디렉토리 경로", placeholder="/home/user/projects")
    request_input = gr.Textbox(
        label="요청", placeholder="Create a Flutter login screen 또는 Run the app")
    project_dir_state = gr.State(value=None)

    submit_button = gr.Button("요청 처리")

    status_output = gr.Textbox(label="상태 메시지", lines=5)
    code_output = gr.Textbox(label="생성/수정된 코드", lines=10)

    submit_button.click(
        fn=gradio_interface,
        inputs=[directory_input, request_input, project_dir_state],
        outputs=[status_output, code_output, project_dir_state]
    )

app.launch()
