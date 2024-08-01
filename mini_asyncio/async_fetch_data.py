#https://medium.com/@moraneus/mastering-pythons-asyncio-a-practical-guide-0a673265cf04
import aiohttp
import asyncio
import time

async def fetch_async(url, session):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        page1 = asyncio.create_task(fetch_async('http://nba.hupu.com', session))
        page2 = asyncio.create_task(fetch_async('http://www.shenlanxueyuan.com', session))
        await asyncio.gather(page1, page2)

start_time = time.time()
asyncio.run(main())
print(f"Done in {time.time() - start_time} seconds")

# Output: Done in 0.2990539073944092 seconds
