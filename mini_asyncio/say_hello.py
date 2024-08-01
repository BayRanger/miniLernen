#https://medium.com/@moraneus/mastering-pythons-asyncio-a-practical-guide-0a673265cf04


import asyncio



async def say_hello_async():
    await asyncio.sleep(2)  # Simulates waiting for 2 seconds
    print("Hello, Async World!")

async def do_something_else():
    print("Starting another task...")
    await asyncio.sleep(1)  # Simulates doing something else for 1 second
    print("Finished another task!")

async def main():
    # Schedule both tasks to run concurrently
    await asyncio.gather(
        say_hello_async(),
        do_something_else(),
    )

asyncio.run(main())
