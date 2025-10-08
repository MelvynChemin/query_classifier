from os import listdir
from os.path import isfile, join
import json

# List all files in ./data
onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]
print("Files found:", onlyfiles)

with open("data/combined_data.jsonl", "w") as outfile:
    for f in onlyfiles:
        with open(join("data", f), "r") as infile:
            for line in infile:
                obj = json.loads(line)
                outfile.write(json.dumps(obj) + "\n")
