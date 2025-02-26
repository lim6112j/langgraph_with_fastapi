# airbnb data
from datasets import load_dataset
import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv("./data/locations.csv")
def draw_route(start, destination):

    fig = go.Figure().add_trace(go.Scattermapbox(
        mode='lines',
        lon = [start[0], destination[0]],
        lat = [start[1], destination[1]],
        line_color='green',
        name='routes calculated'
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=37.497467,
                lon=127.027458
            ),
            pitch=0,
            zoom=15
        ),
    )
    return fig

def filter_map():
    names = df["name"].tolist()
    lons = df["longitude"].tolist()
    lats = df["latitude"].tolist()
    loc_list = [(names[i], (lons[i], lats[i])) for i in range(0, len(names))]
    # map figure
    fig = go.Figure(go.Scattermapbox(
        customdata=loc_list,
        lat=lats,
        lon=lons,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6
        ),
        hoverinfo="text",
        hovertemplate='<b>Name</b>: %{customdata[0]}<br><b>Loc</b>: $%{customdata[1]}'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=37.497467,
                lon=127.027458
            ),
            pitch=0,
            zoom=13
        ),
    )
    return fig

def draw_route_list_closure(g):
    graph = g
    def draw_route_list():
        # loc_list : routes
        config = {"configurable": {"thread_id": "thread_1"}}
        state = graph.get_state(config)

        try:
            routes = state.values['routes']
            print(f"\nstate => {routes}\n")
            loc_list = routes[0]
            print(f"routes[0] : {loc_list}")
            lons = []
            lats = []
            for leg in loc_list['legs']:
                for step in leg['steps']:
                    #           print(f"maneuver data : {step[''maneuver']}")
                    location = (step['maneuver'])['location']
                    lons.append(location[0])
                    lats.append(location[1])
#                    print(f"\nlats from routes data => {lats}\n")
            fig = go.Figure().add_trace(go.Scattermapbox(
                mode='lines',
                lon = lons,
                lat = lats,
                line_color='green',
                name='routes calculated'
            ))
            fig.update_layout(
                mapbox_style="open-street-map",
                hovermode='closest',
                mapbox=dict(
                    bearing=0,
                    center=go.layout.mapbox.Center(
                        lat=37.497467,
                        lon=127.027458
                    ),
                    pitch=0,
                    zoom=13
                ),
            )
            return fig
        except:
            print("routes not ready")

    return draw_route_list()

def get_menu_list_closure(g):
    graph = g
    def get_menus():
        config = {"configurable": {"thread_id": "thread_1"}}
        state = graph.get_state(config)
        try:
            menus = state.values['menus']
#            print(f"state[menus] : {menus}")
            return menus
        except:
            print("menus not ready")
    return get_menus()

from PIL import Image
import requests
from io import BytesIO

def get_image_from_url(img_url):
    res = requests.get(img_url)
    img = Image.open(BytesIO(res.content))
    return img
def get_chart_data_closure(g):
    graph = g
    def get_data():
        config = {"configurable": {"thread_id": "thread_1"}}
        state = graph.get_state(config)
        try:
            data = state.values['chart_data']
#            print(f"state[menus] : {menus}")
            return data
        except:
            print("state data not ready")
    return get_data()
