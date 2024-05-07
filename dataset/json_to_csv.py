# in theis code I am taking info that I need from our dataset and make dataframe from it. Saving the results in csv format

import pandas as pd
import os
import json

json_file_path = 'arxiv-metadata-oai-snapshot.json'

with open(json_file_path, 'r') as file:
    json_data_list = [json.loads(next(file)) for _ in range(200)]

data = {
    "id": [entry.get("id", "") for entry in json_data_list],
    "author": [entry.get("authors", "") for entry in json_data_list],
    "title": [entry.get("title", "") for entry in json_data_list],
    "abstract": [entry.get("abstract", "") for entry in json_data_list],
}


df = pd.DataFrame(data)
print(df)
df.to_csv(os.path.join(os.getcwd(), 'arxiv_metadata.csv'), index=False)
