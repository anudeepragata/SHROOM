import json
 
# Opening JSON file
f = open('SHROOM_unlabeled-training-data-v2/train.model-agnostic.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
parsed_data = []
for i in data:
    datum = {}
    for key in i:
        datum[key] = i[key]
    parsed_data.append(datum)
 
# Closing file
f.close()