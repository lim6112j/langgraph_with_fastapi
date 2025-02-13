from typing import Union
from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from agent import run_agent
app = FastAPI()
class Message(BaseModel):
    message: str
    session_id: str
    thread_id: Union[str, None] = None

@app.post("/messages", status_code=status.HTTP_201_CREATED)
async def create_message(message: Message):
    response = run_agent(message.message, message.session_id, message.thread_id)
    return {"response": response}

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None
items = [{1: {"name": "foo", "price": 5.0}}, {2: {"name": "bar", "price": 3.0}}, {3: {"name": "baz", "price": 9.5}}]

@app.get("/")
def read_root():
    run_agent()
    return {"hello": "world"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def upsert_item(item_id: int, item: Item):
    if item_id in items:
        new_item = items[item_id]
        new_item["name"] = item.name
        new_item["price"] = item.price
        return new_item
    else:
        item = {"name": item.name, "price": item.price}
        items[item_id] = item
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=item)


@app.post("/items/", response_model=Item, summary="create an item")
async def create_item(item: Item):
    """
    Create an item with all the information:

    - **name**: each name must be unique
    - **price**: required field
    - **is_offer**: optional
    \f
    :param item: User input
    """
    return item
