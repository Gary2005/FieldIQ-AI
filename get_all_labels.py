path = "game_example"
label_path = f"{path}/Labels-v2.json"
frame_informati_path = f"{path}/output.json"

import json
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

label_json = json.load(open(label_path, "r"))
frame_information_json = json.load(open(frame_informati_path, "r"))

print(len(label_json["annotations"]))

label = set()
for i in tqdm(range(len(label_json["annotations"]))):
    label.add(label_json["annotations"][i]["label"])

print("Total labels: ", len(label))
print("Labels: ", label)