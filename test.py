import requests

url = "http://127.0.0.1:5000/analyze"
files = {"video": open("test_video.mp4", "rb")}

res = requests.post(url, files=files)
print(res.json())
