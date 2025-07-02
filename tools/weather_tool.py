import python_weather
import asyncio

async def get_weather(city: str) -> str:
    try:
        async with python_weather.Client() as client:
            weather = await client.get(city)
            current = weather.current
            return f"The weather in {city} is {current.temperature}°C with {current.sky_text}."
    except Exception as e:
        return f"Couldn't fetch weather for {city}: {str(e)}"
