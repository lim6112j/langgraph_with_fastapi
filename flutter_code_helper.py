import os
import shutil
import re
import subprocess
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from dotenv import load_dotenv
import getpass
import json
load_dotenv()
# OpenAI API 키 설정
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Api key for OPENAI")

# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0.2, max_tokens=4096)

# 대화 메모리 초기화
memory = ConversationBufferMemory()

# Flutter 작업을 위한 프롬프트 템플릿 (BLoC 기반)
flutter_prompt = PromptTemplate(
    input_variables=["request", "history", "filename"],
    template="""
    You are an expert Flutter developer familiar with the BLoC pattern. Based on the user's request, conversation history, and the specified filename, perform the requested task. Use the `flutter_bloc` package for state management when generating code. Include necessary imports, widgets, BLoC setup (Bloc, Event, State), and basic styling. Assume the `flutter_bloc` package is already added to `pubspec.yaml`. If generating code, provide it in a markdown code block (```dart) for the file named '{filename}'. If analyzing or modifying code, reference the previously generated code from the history. Ensure the code is a complete, self-contained widget or class suitable for a separate file, not embedded in `main.dart`.

    Conversation history: {history}

    User's request: {request}

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

    If the request implies creating or modifying a specific screen (e.g., "login screen"), suggest a filename like `lib/screens/<screen_name>.dart`. If no specific file is implied, use `lib/main.dart`.

    Conversation history: {history}

    User's request: {request}

    Output a JSON object with:
    - "action": One of ["create_project", "process_existing", "run_app", "other"]
    - "processed_request": The request to pass to the selected action
    - "filename": The target filename (e.g., "lib/screens/login_screen.dart" or "lib/main.dart")
    """
)

# LangChain 체인 생성
flutter_chain = LLMChain(llm=llm, prompt=flutter_prompt)
chatbot_chain = LLMChain(llm=llm, prompt=chatbot_prompt)

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

# LangGraph 노드 정의


def chatbot_node(state: GraphState) -> GraphState:
    """사용자 요청을 분석하는 챗봇 노드"""
    try:
        response = chatbot_chain.run(
            request=state["request"], history=memory.buffer)
        response_json = json.loads(response)  # 안전한 JSON 파싱
        state["action"] = response_json["action"]
        state["processed_request"] = response_json["processed_request"]
        state["filename"] = response_json["filename"]
        return state
    except Exception as e:
        state["status"] = f"Error in chatbot: {e}"
        state["action"] = "other"
        state["filename"] = "lib/main.dart"
        return state


def create_project_node(state: GraphState) -> GraphState:
    """Flutter 프로젝트 생성 노드"""
    try:
        directory = os.path.abspath(state["directory"])
        project_dir = os.path.join(directory, "my_flutter_app")
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)

        if not shutil.which("flutter"):
            raise Exception("Flutter CLI가 설치되어 있지 않습니다.")

        os.system(f"flutter create --quiet {project_dir}")

        pubspec_path = os.path.join(project_dir, "pubspec.yaml")
        with open(pubspec_path, "r") as f:
            pubspec_content = f.read()

        if "flutter_bloc" not in pubspec_content:
            with open(pubspec_path, "a") as f:
                f.write("\n  flutter_bloc: ^8.1.3\n")

        os.system(f"cd {project_dir} && flutter pub get")

        # 요청된 파일 생성
        response = flutter_chain.invoke({
            "request": state["processed_request"],
            "history": memory.buffer,
            "filename": state["filename"]
        })["text"]
        code_match = re.search(r"```dart\n([\s\S]*?)\n```", response)
        dart_code = code_match.group(1) if code_match else response

        # 파일 경로 설정
        file_path = os.path.join(project_dir, state["filename"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(dart_code)

        # main.dart 기본 코드 생성 (필요 시)
        main_dart_path = os.path.join(project_dir, "lib", "main.dart")
        if state["filename"] != "lib/main.dart":
            class_name = os.path.basename(
                state["filename"]).replace(".dart", "").capitalize()
            main_code = """
import 'package:flutter/material.dart';
import '{relative_import}';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: {class_name}(),
    );
  }
}
""".format(
                relative_import=state["filename"].replace(
                    "lib/", "").replace(".dart", ""),
                class_name=class_name
            )
            with open(main_dart_path, "w") as f:
                f.write(main_code)

        state["status"] = f"Flutter 프로젝트가 {project_dir}에 생성되었습니다. 파일: {state['filename']}"
        state["code"] = dart_code
        state["project_dir"] = project_dir
        return state
    except Exception as e:
        state["status"] = f"Error: {e}"
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
            "filename": state["filename"]
        })["text"]
        code_match = re.search(r"```dart\n([\s\S]*?)\n```", response)
        if code_match:
            dart_code = code_match.group(1)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(dart_code)
            state["status"] = f"{state['filename']}가 업데이트되었습니다: {file_path}"
            state["code"] = dart_code
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
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode == 0:
            state["status"] = f"앱이 성공적으로 실행되었습니다.\nOutput:\n{stdout}"
        else:
            state["status"] = f"앱 실행 실패:\nError:\n{stderr}"

        file_path = os.path.join(
            project_dir, state.get("filename", "lib/main.dart"))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                state["code"] = f.read()
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
        code=""
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
