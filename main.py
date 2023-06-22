from pydantic import BaseModel
import torch
import os
torch.hub.set_dir('/persistent-storage/detoxify')
model = torch.hub.load('unitaryai/detoxify','unbiased_toxic_roberta')
from detoxify import Detoxify

class Item(BaseModel):
    text: str


def predict(item, run_id, logger):
    item = Item(**item)
    results = Detoxify('unbiased',device='cuda').predict(item.text)
    data = {k: float(v) for k, v in result.items()}
    return data
            
