import requests
import json
def download_video(_id: str, quality: str = "720p") -> str or None:
    res = requests.post("http://localhost:5000/download/"+quality, json={
        "url":"https://www.youtube.com/watch?v=" + _id
    })
    return json.loads(res.content).get("file_path")