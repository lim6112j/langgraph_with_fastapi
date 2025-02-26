from langchain_core.tools import InjectedToolCallId, tool
from typing import Annotated, Union, Dict, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import json
import pandas as pd
import os
import http.client

class State(TypedDict):
    messages: Annotated[list, add_messages]
    menus: list
    routes: list
    chart_data: list

@tool
def get_routes(state: State, start: str, destination: str,  tool_call_id: Annotated[str, InjectedToolCallId]):
    """ Get route information from start to destination.
    after gather start and destination from user,
    call this function for route data
    Args:
       start (str): start location name
       destination (str): destination location name
    """
    df = pd.read_csv("./data/locations.csv")
    names = df["name"].tolist()
    lons = df["longitude"].tolist()
    lats = df["latitude"].tolist()
    loc_dict = {names[i]: (lons[i], lats[i]) for i in range(0, len(names))}
    start_loc = str(loc_dict[start][0]) + ',' + str(loc_dict[start][1])
    destination_loc = str(loc_dict[destination][0]) + ',' + str(loc_dict[destination][1])
    locations = start_loc + ';' + destination_loc
#    print(locations)


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
@tool
def get_menus(state: State,tool_call_id: Annotated[str, InjectedToolCallId]):
    """getting menus of restaurant
    """
    df_menus = pd.read_csv("./data/menus.csv")
    dict = df_menus.to_json(orient="records")
    menus = f"{dict}"
    return Command(
         update={
             "menus": menus,
             "messages": [
                 ToolMessage(menus, tool_call_id=tool_call_id)
             ],
         }
    )
@tool
def get_data_from_site(state: State, keyword: str):
    """get web data from website
    Args: keyword(str): keyword you will search for
    """
    try:
        import urllib.parse
        safe_keyword = urllib.parse.quote(keyword)
        headers={'user-agent': "Emacs Restclient"}
        conn = http.client.HTTPSConnection("namu.wiki")
        conn.request("GET", "/Search?q=" + safe_keyword, headers=headers)
        res = conn.getresponse()
        res_bytes = res.read()
        html_str = res_bytes.decode('utf8').replace("'", '"')
        print(f"got data from namuwiki => {html_str}")
        conn.close()
    except TypeError as ex:
        print(type(ex))
        print(ex.args)
        print(ex)
        html_str = "not found"
    return f"{html_str}"

@tool
def get_dashboard_info(state: State):
    """get dashboard data"""
    df = pd.read_csv("./data/dashboard.csv")
    dict = df.to_json(orient="records")
    return f"{dict}"
@tool
def get_chart_data(state: State, data: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """draw chart"""
    return Command(
        update={
            "chart_data": data,
             "messages": [
                 ToolMessage(data, tool_call_id=tool_call_id)
             ],
        }
    )
@tool
def get_postgresql_data(state: State, query: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """get postgreql data with given query"""
    data = "test"
    return f"{data}"
