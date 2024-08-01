import requests
import time

start_time = time.time()

def fetch(url):
    return requests.get(url).text

page1 = fetch('https://nba.hupu.com')
page2 = fetch('https://www.shenlanxueyuan.com/')


print(f"Done in {time.time() - start_time} seconds")

# Output: Done in 0.6225857734680176 seconds
