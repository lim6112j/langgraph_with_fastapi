# langgrph_with_fastapi

brew install portaudio

run docker for osrm

source .venv/bin/activate

* run server.py

fastapi dev server.py

use restapi 강남역, 양재역

* run audio input

python whisper.py

+ run audio with ui

python tools_with_ui.py

* run langgraph with ui

gradio agent_with_ui.py

* run routing agent
gradio agent_routing.py

* run menu-pan agent
gradio agent_menu.py

* run crawler
gradio agent_crawler.py

* dashboard
gradio agent_dashboard.py

# how to make new tool

add new template in ./messeages/messages.py

add new function in ./custom_tool/tools.py

add new function in ./helper/func.py

make new agent in / : e.g. agent_new_xxx.py

* flutter code generation

사용자가 "Gradio 코드 내의 모든 메서드가 LangGraph 노드 도구로 구현되어야 한다"고 의도한 것을 이제 이해했습니다. 즉, Flutter 코드 자체가 아니라, Gradio 애플리케이션의 Python 메서드(`create_flutter_project`, `process_existing_project`, `run_flutter_app` 등)를 LangGraph의 노드 도구로 변환하라는 의미입니다. 이를 위해 LangGraph를 사용해 워크플로우를 그래프 구조로 정의하고, 각 기능을 독립적인 노드로 구현하겠습니다.

---

### 요구사항
- Gradio UI에서 Flutter 프로젝트 생성, 분석, 수정, 실행 요청 처리.
- Gradio 코드의 메서드가 LangGraph 노드 도구로 구성.
- BLoC 기반 Flutter 코드 생성 유지.
- 대화 메모리 및 앱 실행 기능 포함.

---

### 코드

```python
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

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # 실제 키로 교체

# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0.2, max_tokens=4096)

# 대화 메모리 초기화
memory = ConversationBufferMemory()

# Flutter 작업을 위한 프롬프트 템플릿 (BLoC 기반)
flutter_prompt = PromptTemplate(
    input_variables=["request", "history"],
    template="""
    You are an expert Flutter developer familiar with the BLoC pattern. Based on the user's request and the conversation history, perform the requested task. Use the `flutter_bloc` package for state management when generating code. Include necessary imports, widgets, BLoC setup (Bloc, Event, State), and basic styling. Assume the `flutter_bloc` package is already added to `pubspec.yaml`. If generating code, provide it in a markdown code block (```dart). If analyzing or modifying code, reference the previously generated code from the history.

    Conversation history: {history}

    User's request: {request}

    Output the result (code in ```dart block if applicable, otherwise plain text). If the request is ambiguous, make reasonable assumptions.
    """
)

# LangChain 체인 생성 (메모리 포함)
flutter_chain = LLMChain(llm=llm, prompt=flutter_prompt, memory=memory)

# LangGraph 상태 정의
class GraphState(Dict[str, Any]):
    directory: str
    request: str
    project_dir: str
    status: str
    code: str

# LangGraph 노드 정의
def check_project_exists(state: GraphState) -> GraphState:
    """프로젝트 존재 여부 확인 노드"""
    if state.get("project_dir") and os.path.exists(state["project_dir"]):
        return state
    elif "create" in state["request"].lower() or "프로젝트" in state["request"]:
        state["status"] = "프로젝트 생성 요청 감지"
        return state
    else:
        state["status"] = "Error: 먼저 Flutter 프로젝트를 생성하세요."
        state["code"] = None
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

        response = flutter_chain.run(request=state["request"], history=memory.buffer)
        code_match = re.search(r"```dart\n([\s\S]*?)\n```", response)
        dart_code = code_match.group(1) if code_match else response

        main_dart_path = os.path.join(project_dir, "lib", "main.dart")
        with open(main_dart_path, "w") as f:
            f.write(dart_code)

        state["status"] = f"Flutter 프로젝트가 {project_dir}에 생성되었습니다."
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
        main_dart_path = os.path.join(project_dir, "lib", "main.dart")
        with open(main_dart_path, "r") as f:
            existing_code = f.read()

        memory.chat_memory.add_user_message(f"Previous main.dart code:\n```dart\n{existing_code}\n```")

        response = flutter_chain.run(request=state["request"], history=memory.buffer)
        code_match = re.search(r"```dart\n([\s\S]*?)\n```", response)
        if code_match:
            dart_code = code_match.group(1)
            with open(main_dart_path, "w") as f:
                f.write(dart_code)
            state["status"] = f"main.dart가 업데이트되었습니다: {main_dart_path}"
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

        main_dart_path = os.path.join(project_dir, "lib", "main.dart")
        with open(main_dart_path, "r") as f:
            state["code"] = f.read()
        return state
    except subprocess.TimeoutExpired:
        process.kill()
        state["status"] = "Error: 앱 실행이 30초 내에 완료되지 않았습니다."
        return state
    except Exception as e:
        state["status"] = f"Error: {e}"
        return state

# LangGraph 워크플로우 정의
workflow = StateGraph(GraphState)

workflow.add_node("check_project", check_project_exists)
workflow.add_node("create_project", create_project_node)
workflow.add_node("process_existing", process_existing_project_node)
workflow.add_node("run_app", run_app_node)

workflow.set_entry_point("check_project")
workflow.add_conditional_edges(
    "check_project",
    lambda state: "create_project" if "create" in state["request"].lower() or "프로젝트" in state["request"] else "process_existing" if not state["status"].startswith("Error") else "end",
    {"create_project": "create_project", "process_existing": "process_existing", "end": END}
)
workflow.add_edge("create_project", END)
workflow.add_conditional_edges(
    "process_existing",
    lambda state: "run_app" if "run" in state["request"].lower() or "실행" in state["request"] else "end",
    {"run_app": "run_app", "end": END}
)
workflow.add_edge("run_app", END)

app_graph = workflow.compile()

# Gradio 인터페이스 처리 함수
def gradio_interface(directory, request, project_dir_state):
    initial_state = GraphState(
        directory=directory,
        request=request,
        project_dir=project_dir_state,
        status="",
        code=""
    )
    result = app_graph.invoke(initial_state)
    return result["status"], result["code"], result["project_dir"]

# Gradio 앱 실행
with gr.Blocks(title="Flutter Project Manager with BLoC") as app:
    gr.Markdown("# Flutter 프로젝트 관리기 (BLoC 기반)")
    gr.Markdown("프로젝트 생성, 코드 분석, 수정, 실행 등을 요청할 수 있습니다.")

    directory_input = gr.Textbox(label="디렉토리 경로", placeholder="/home/user/projects")
    request_input = gr.Textbox(label="요청", placeholder="Create a Flutter login screen 또는 Run the app")
    project_dir_state = gr.State(value=None)

    submit_button = gr.Button("요청 처리")

    status_output = gr.Textbox(label="상태 메시지", lines=5)
    code_output = gr.Textbox(label="main.dart 코드", lines=10)

    submit_button.click(
        fn=gradio_interface,
        inputs=[directory_input, request_input, project_dir_state],
        outputs=[status_output, code_output, project_dir_state]
    )

app.launch()

```

---

### 코드 설명

1. **LangGraph 통합**:
   - `langgraph.graph.StateGraph`로 워크플로우 정의.
   - 상태는 `GraphState` 클래스로 관리(딕셔너리 기반).

2. **노드 도구**:
   - `check_project_exists`: 프로젝트 존재 여부 확인 및 다음 단계 결정.
   - `create_project_node`: Flutter 프로젝트 생성.
   - `process_existing_project_node`: 기존 프로젝트에서 코드 분석/수정.
   - `run_app_node`: Flutter 앱 실행.

3. **그래프 흐름**:
   - `check_project` → (`create_project` 또는 `process_existing`) → (`run_app` 또는 종료).
   - 조건부 엣지로 요청 타입에 따라 경로 분기.

4. **Gradio UI**:
   - 입력을 받아 LangGraph 워크플로우 실행.
   - 상태와 코드를 UI에 반영.

---

### 실행 예시

#### 1. 프로젝트 생성
- **입력**:
  - 디렉토리: `/home/user/projects`
  - 요청: `Create a Flutter login screen with email and password fields`
- **출력**:
  - 상태 메시지: `Flutter 프로젝트가 /home/user/projects/my_flutter_app에 생성되었습니다.`
  - 코드: (BLoC 기반 로그인 화면)

#### 2. 앱 실행
- **입력**:
  - 요청: `Run the app`
- **출력**:
  - 상태 메시지: `앱이 성공적으로 실행되었습니다.\nOutput:\nLaunching lib/main.dart...`
  - 코드: (현재 `main.dart`)

#### 3. 코드 분석
- **입력**:
  - 요청: `Analyze the code`
- **출력**:
  - 상태 메시지: `The code uses BLoC for state management...`
  - 코드: (분석 결과)

---

### 실행 방법
1. **준비**:
   - Flutter SDK 설치.
   - 에뮬레이터 준비.
   - OpenAI API 키 입력.
   - 의존성 설치:
     ```bash
     pip install gradio langchain langchain-openai langgraph
     ```
2. **실행**:
   ```bash
   python your_script.py
   ```

---

### 주의사항
1. **LangGraph 의존성**: `pip install langgraph` 필요.
2. **에뮬레이터**: `-d emulator`는 환경에 맞게 조정.
3. **단일 파일**: Flutter 코드는 `main.dart`에 집중.

---

### 개선 제안
1. **노드 확장**: 더 세분화된 노드 추가(예: `install_dependencies`).
2. **실시간 로그**: `run_app_node`에서 스트리밍 출력.
3. **그래프 시각화**: LangGraph 워크플로우를 UI에 표시.

의도에 맞는지 확인해 주시고, 추가 요청이 있으면 말씀해 주세요!


in previous gradio app, make chatbot node and combine tools with the chatbot node for request processing

사용자가 이전 Gradio 기반 Flutter 프로젝트 관리 애플리케이션에 챗봇 노드를 추가하고, 기존 LangGraph 노드 도구들(create_project_node, process_existing_project_node, run_app_node)을 이 챗봇 노드와 통합하여 요청을 처리하도록 요청했습니다. 이를 위해 LangGraph 워크플로우에 챗봇 노드를 추가하고, 사용자의 자연어 요청을 해석해 적절한 도구를 호출하도록 구현하겠습니다. 챗봇 노드는 LangChain의 LLM을 사용해 요청을 분석하고, LangGraph의 상태를 업데이트하며, 기존 노드들과 조합해 작업을 수행합니다.

별로 파일로 요청한 내용을 생성하라고 지시했는데 계속 main.dart만 수정하는 문제가 있음.

사용자가 요청한 내용을 별도의 파일로 생성하도록 지시했음에도 불구하고 코드가 계속 main.dart만 수정하고 있다고 지적하셨습니다. 사용자의 의도를 반영하여, 요청에 따라 새로운 Dart 파일(예: login_screen.dart, home_screen.dart 등)을 생성하거나 수정하도록 코드를 수정하겠습니다. 이를 위해 LangGraph 워크플로우와 챗봇 노드를 유지하면서, 요청을 분석해 적절한 파일명을 결정하고, 새로운 파일을 생성하거나 기존 파일을 수정하는 로직을 추가합니다. 또한, Gradio UI와 BLoC 기반 Flutter 코드 생성, 앱 실행 기능은 그대로 유지합니다.

이메일, 패스워드를 필드로 하는 로그인 화면이 포함된 flutter app을 만들라고 지시했는데, 에러가 발생했어
Error: One input key expected got ['filename', 'request']

사용자가 "이메일, 패스워드를 필드로 하는 로그인 화면이 포함된 Flutter 앱을 만들라고 지시했는데 에러가 발생했다"고 하셨으며, 구체적으로 Error: One input key expected got ['filename', 'request']라는 에러를 언급하셨습니다. 이 에러는 LangChain의 LLMChain이 단일 입력 키를 기대하는데, flutter_chain 호출 시 filename과 request 두 개의 입력을 전달하면서 발생한 것으로 보입니다. 이는 flutter_prompt 템플릿이 filename, request, history를 모두 처리하도록 설계되었음에도, LLMChain 실행 시 입력 처리 로직이 잘못된 경우 발생할 수 있습니다.


Error: '\n  runApp(MyApp());\n'

사용자가 Flutter 앱 생성 요청("이메일, 패스워드를 필드로 하는 로그인 화면")을 처리하는 과정에서 Error: '\n  runApp(MyApp());\n'라는 에러가 발생했다고 언급하셨습니다. 이 에러는 main.dart 파일의 코드가 잘못된 형식으로 생성되었거나, runApp(MyApp()); 호출 주변에 예상치 못한 공백이나 들여쓰기 문제로 인해 발생한 것으로 보입니다. 특히, 이전 코드에서 main.dart를 생성할 때 문자열 포맷팅(.format())이 올바르게 처리되지 않았거나, 파일 내용이 잘못된 구조로 저장되었을 가능성이 있습니다.
