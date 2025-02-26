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
