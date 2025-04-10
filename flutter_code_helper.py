import os
import shutil
import re
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import getpass
from langchain.memory import ConversationBufferMemory
load_dotenv()

# OpenAI API 키 설정
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Api key for OPENAI")

# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0.2, max_tokens=4096)

# 대화 메모리 초기화
memory = ConversationBufferMemory()

# Flutter 작업을 위한 프롬프트 템플릿
flutter_prompt = PromptTemplate(
    input_variables=["request", "history"],
    template="""
    You are an expert Flutter developer familiar with the BLoC pattern. Based on the user's request and the conversation history, perform the requested task. Use the `flutter_bloc` package for state management when generating code. Include necessary imports, widgets, and basic styling. If generating code, provide it in a markdown code block (```dart). If analyzing or modifying code, reference the previously generated code from the history.

    Conversation history: {history}

    User's request: {request}

    Output the result (code in ```dart block if applicable, otherwise plain text). If the request is ambiguous, make reasonable assumptions.
    """
)

# LangChain 체인 생성 (메모리 포함)
flutter_chain = LLMChain(llm=llm, prompt=flutter_prompt, memory=memory)

# Flutter 프로젝트 생성 함수
def create_flutter_project(directory, request):
    """
    지정된 디렉토리에 BLoC 기반 Flutter 프로젝트를 생성하고 요청에 따른 코드를 작성합니다.
    """
    try:
        if not directory or not request:
            return "Error: 디렉토리 경로와 요청을 입력하세요.", None, None

        directory = os.path.abspath(directory)
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

        # LangChain으로 코드 생성
        response = flutter_chain.run(request=request, history=memory.buffer)
        code_match = re.search(r"```dart\n([\s\S]*?)\n```", response)
        dart_code = code_match.group(1) if code_match else response

        main_dart_path = os.path.join(project_dir, "lib", "main.dart")
        with open(main_dart_path, "w") as f:
            f.write(dart_code)

        return f"Flutter 프로젝트가 {project_dir}에 생성되었습니다.\n실행: cd {project_dir} && flutter run", dart_code, project_dir
    except Exception as e:
        return f"Error: {e}", None, None

# 기존 프로젝트에서 작업 수행 함수
def process_existing_project(project_dir, request):
    """
    기존 프로젝트의 main.dart를 읽고 요청에 따라 작업을 수행합니다.
    """
    try:
        if not project_dir or not os.path.exists(project_dir):
            return "Error: 유효한 프로젝트 디렉토리를 먼저 생성하세요.", None

        main_dart_path = os.path.join(project_dir, "lib", "main.dart")
        with open(main_dart_path, "r") as f:
            existing_code = f.read()

        # 메모리에 기존 코드 추가
        memory.chat_memory.add_user_message(f"Previous main.dart code:\n```dart\n{existing_code}\n```")

        # LangChain으로 요청 처리
        response = flutter_chain.run(request=request, history=memory.buffer)
        code_match = re.search(r"```dart\n([\s\S]*?)\n```", response)
        if code_match:
            dart_code = code_match.group(1)
            with open(main_dart_path, "w") as f:
                f.write(dart_code)
            return f"main.dart가 업데이트되었습니다: {main_dart_path}", dart_code
        else:
            return response, None
    except Exception as e:
        return f"Error: {e}", None

# Gradio 인터페이스 처리 함수
def gradio_interface(directory, request, project_dir_state):
    """
    Gradio UI에서 입력을 받아 작업을 처리합니다.
    """
    if not project_dir_state:  # 프로젝트가 아직 생성되지 않은 경우
        if "create" in request.lower() or "프로젝트" in request:
            status, code, new_project_dir = create_flutter_project(directory, request)
            return status, code, new_project_dir
        else:
            return "Error: 먼저 Flutter 프로젝트를 생성하세요 (예: 'Create a Flutter login screen').", None, None
    else:  # 기존 프로젝트에서 작업 수행
        status, code = process_existing_project(project_dir_state, request)
        return status, code, project_dir_state

# Gradio 앱 실행
with gr.Blocks(title="Flutter Project Manager with BLoC") as app:
    gr.Markdown("# Flutter 프로젝트 관리기 (BLoC 기반)")
    gr.Markdown("프로젝트 생성 후 코드 분석, 수정 등 추가 요청을 처리할 수 있습니다.")

    # 입력 및 상태 관리
    directory_input = gr.Textbox(label="디렉토리 경로", placeholder="/home/user/projects")
    request_input = gr.Textbox(label="요청", placeholder="Create a Flutter login screen 또는 Analyze the code")
    project_dir_state = gr.State(value=None)  # 현재 프로젝트 디렉토리 상태

    submit_button = gr.Button("요청 처리")

    # 출력
    status_output = gr.Textbox(label="상태 메시지", lines=5)
    code_output = gr.Textbox(label="main.dart 코드", lines=10)

    # 요청 처리 로직
    submit_button.click(
        fn=gradio_interface,
        inputs=[directory_input, request_input, project_dir_state],
        outputs=[status_output, code_output, project_dir_state]
    )

app.launch()
