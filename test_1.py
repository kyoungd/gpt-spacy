import asyncio
import aiohttp
import requests
from random import randint
from time import perf_counter
from collections.abc import AsyncIterable

# The highest Pokemon id
MAX_POKEMON = 898


def get_random_pokemon_name_sync() -> str:
    pokemon_id = randint(1, MAX_POKEMON)
    pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    response = requests.get(pokemon_url)
    pokemon = response.json()
    return str(pokemon["name"])


async def get_random_pokemon_name() -> str:
    async with aiohttp.ClientSession() as session:
        pokemon_id = randint(1, MAX_POKEMON)
        pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
        async with session.get(pokemon_url) as response:
            pokemon = await response.json()
            return str(pokemon["name"])


async def next_pokemon(total: int) -> AsyncIterable[str]:
    for _ in range(total):
        name = await get_random_pokemon_name()
        yield name

async def next_pokemon_concurrent(total: int) -> AsyncIterable[str]:
    tasks = [get_random_pokemon_name() for _ in range(total)]
    for name in await asyncio.gather(*tasks):
        yield name

async def main() -> None:
    time_before = perf_counter()
    names = [name async for name in next_pokemon_concurrent(20)]
    print(f"Total time (asynchronous): {perf_counter() - time_before}")
    print(names)

    # asynchronous call
    time_before = perf_counter()
    result_async = await asyncio.gather(*[get_random_pokemon_name() for _ in range(20)])
    print(f"Total time (asynchronous): {perf_counter() - time_before}")
    print(result_async)

# async def main() -> None:

#     # synchronous call
#     time_before = perf_counter()
#     for _ in range(20):
#         get_random_pokemon_name_sync()
#     print(f"Total time (synchronous): {perf_counter() - time_before}")

#     # asynchronous call
#     time_before = perf_counter()
#     result_async = await asyncio.gather(*[get_random_pokemon_name() for _ in range(20)])
#     print(f"Total time (asynchronous): {perf_counter() - time_before}")
#     print(result_async)


asyncio.run(main())
