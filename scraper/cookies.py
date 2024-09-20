import asyncio
import json

async def main():
  with open('scraper/raw_cookies.json', 'r') as file:
    data = json.load(file)

  result = {}
  for item in data:
    name = item.get("name")
    value = item.get("value")
    if name and value:
      result[name] = value
  
  with open('scraper/twikit_cookies.json', 'w') as file:
    json.dump(result, file, indent=4)

asyncio.run(main())