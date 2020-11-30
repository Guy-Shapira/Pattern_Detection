import json


def decode(o):
    if isinstance(o, str):
        try:
            return int(o)
        except ValueError:
            return "../" + o
    elif isinstance(o, dict):
        return {k: decode(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [decode(v) for v in o]
    else:
        return o


with open("constants.json") as json_file:
    constants = json.load(json_file, object_hook=decode)
    for i in range(len(constants["event_types"])):
        constants["event_types"][i] = constants["event_types"][i][-1]
