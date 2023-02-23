#! /bin/bash python3
import sys
from pytube import YouTube




if len(sys.argv) != 2:
    print("only one arg, url link for video lol")
    sys.exit()
#agr length passed from cli


url = sys.argv[1]
#url is the first arg passed

#720p quality
yt = YouTube(url)
stream = yt.streams.get_by_resolution("720p")


video_title = yt.title # formatting for print
filename = f"{video_title}.mp4"


print(f"Downloading '{video_title}' in 720p...")
stream.download(filename=filename)

#downloads and saves in repo

print("Download complete!")