from glob import glob

colors = ["green", "red", "yellow", "blue"]

csv_header = "R,G,B,O,Y,B\n"

compiled_green = csv_header
compiled_yellow = csv_header
compiled_blue = csv_header
compiled_red = csv_header
csv_files = glob("Test_Data/*.csv")
for i in filter(lambda x: "green" in x, csv_files):
    with open(i, "r") as f:
        compiled_green += f.read()
with open("compiled_green.csv", "w") as f:
    f.write(compiled_green)

for i in filter(lambda x: "yellow" in x, csv_files):
    with open(i, "r") as f:
        compiled_yellow += f.read()
with open("compiled_yellow.csv", "w") as f:
    f.write(compiled_yellow)

for i in filter(lambda x: "blue" in x, csv_files):
    with open(i, "r") as f:
        compiled_blue += f.read()
with open("compiled_blue.csv", "w") as f:
    f.write(compiled_blue)

for i in filter(lambda x: "red" in x, csv_files):
    with open(i, "r") as f:
        compiled_red += f.read()
with open("compiled_red.csv", "w") as f:
    f.write(compiled_red)