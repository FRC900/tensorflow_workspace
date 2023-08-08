import os


inital_folder = "combined_88_test"

folders_to_compare = ["/home/ubuntu/tensorflow_workspace/2022Game/data/videos", 
                      "/home/ubuntu/tensorflow_workspace/2020Game/data/videos"]


files = os.listdir(inital_folder)

for file in files:
    for folder in folders_to_compare:
        if file in os.listdir(folder):
            print("Removing " + file)
            os.remove(inital_folder + "/" + file)
            
