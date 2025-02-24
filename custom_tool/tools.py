from langchain_core.tools import InjectedToolCallId, tool
from typing import Annotated, Union, Dict, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import json
import pandas as pd
import os
df = pd.read_csv("./data/locations.csv")
names = df["name"].tolist()
lons = df["longitude"].tolist()
lats = df["latitude"].tolist()
loc_dict = {names[i]: (lons[i], lats[i]) for i in range(0, len(names))}

class State(TypedDict):
    messages: Annotated[list, add_messages]
    routes: list


@tool
def get_routes(state: State, start: str, destination: str,  tool_call_id: Annotated[str, InjectedToolCallId]):
    """ Get route information from start to destination.
    after gather start and destination from user,
    call this function for route data
    Args:
       start (str): start location name
       destination (str): destination location name
    """
    start_loc = str(loc_dict[start][0]) + ',' + str(loc_dict[start][1])
    destination_loc = str(loc_dict[destination][0]) + ',' + str(loc_dict[destination][1])
    locations = start_loc + ';' + destination_loc
#    print(locations)

    import http.client
    OSRM_API = os.getenv("OSRM_API")
    # http://localhost:5001/route/v1/driving/127.919323,36.809656;128.080629,36.699223?steps=true
    conn = http.client.HTTPConnection(OSRM_API, 5001)
    conn.request("GET", "/route/v1/driving/" + locations + "?steps=true")
    res = conn.getresponse()
    routes_bytes = res.read()
    conn.close()
#    print(f"osrm response routes data {routes}")
    if not routes_bytes:
        return "No routes found."
    # response = "\n".join([f"{route['id']}: {route['title']}" for route in routes])

    json_str = routes_bytes.decode('utf8').replace("'", '"')
    json_routes = json.loads(json_str)
    routes = json_routes['routes']

#    print(json_routes['routes'])
#    print(f"\nlength of routes : {len(json_routes['routes'])}")
#    fig = draw_route(loc_dict[start], loc_dict[destination])
#    fig.show()
#    return f"Available routes:\n{routes}"
    return Command(
         update={
             "routes": routes,
             "messages": [
                 ToolMessage(routes_bytes, tool_call_id=tool_call_id)
             ],
         }
    )
