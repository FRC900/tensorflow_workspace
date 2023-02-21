#! /bin/bash python3
import sys
from pytube import YouTube




if len(sys.argv) != 2:
    print("only one arg, url link for video lol")
    sys.exit()

url = sys.argv[1]



#720p stream
yt = YouTube(url)
stream = yt.streams.get_by_resolution("720p")


video_title = yt.title
filename = f"{video_title}.mp4"


print(f"Downloading '{video_title}' in 720p...")
stream.download(filename=filename)
print("Download complete!")