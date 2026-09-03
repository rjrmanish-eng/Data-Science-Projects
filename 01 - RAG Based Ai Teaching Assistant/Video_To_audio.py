# Convert the Videos into Mp3
import os 
import subprocess


files = os.listdir('./downloads')
for file in files:
    # Split The Name
    title = file.split('-')[0].split(' ｜ ')[0]
    
    tutorial_number = file.split("Tutorial #")[1].split(".")[0]
    # print(file)
    print(tutorial_number,title)
    subprocess.run(["ffmpeg","-i",f"downloads/{file}",f"audios/{tutorial_number}_{title}.mp3"])
  
    

 