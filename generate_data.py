import json
import random
from datetime import datetime, timedelta

random.seed(42)

LOCATIONS = [
    # International Cities
    "New York", "London", "Tokyo", "Paris", "Sydney", "Berlin", "Mumbai", "Dubai", "Singapore", 
    "Los Angeles", "Chicago", "Toronto", "Miami", "Seattle", "Boston", "San Francisco", 
    "Barcelona", "Rome", "Amsterdam", "Stockholm",
    # Pakistani Cities
    "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar", 
    "Quetta", "Sialkot", "Hyderabad", "Gujranwala", "Sargodha", "Bahawalpur", " Abbottabad",
    "Mardan", "Gujrat", "Kasur", "Larkana", "Rahim Yar Khan", "Jhang"
]
UNITS_C = ["C", "F"]
UNITS_CONVERT = ["meters", "feet", "kilograms", "pounds", "celsius", "fahrenheit", "liters", "gallons", "km", "miles", "inches", "yards", "grams", "ounces", "tons", "kg", "cm", "mm"]
ISO_CODES = ["USD", "EUR", "GBP", "JPY", "INR", "AUD", "CAD", "CHF", "CNY", "SGD", "KRW", "BRL", "MXN", "RUB", "ZAR", "PKR"]

def generate_date():
    days_ahead = random.randint(0, 60)
    return (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

def create_example(prompt, tool, args, is_refusal=False):
    return {"prompt": prompt, "tool": tool, "args": args, "is_refusal": is_refusal}

def generate_weather_prompts(location):
    templates = [
        f"What's the weather in {location}?",
        f"Tell me the weather for {location}",
        f"Weather forecast for {location}",
        f"How's the weather in {location}",
        f"Is it sunny in {location}?",
        f"weather {location}",
        f"wthr in {location}",
        f"wat is the wether in {location}",
        f"मौसम {location} में क्या है",  # Hindi
        f"el clima en {location}",  # Spanish
        f" الطقس في {location}",  # Arabic
        f"کراچی میں موسم کیسے ہے",  # Urdu - Karachi
        f"لاہور میں weather",  # Urdu - Lahore
        f"اسلام آباد میں temperature",  # Urdu - Islamabad
    ]
    return random.choice(templates)

def generate_calendar_list_prompts(date):
    templates = [
        f"Show my calendar for {date}",
        f"What's on my calendar for {date}?",
        f"List events on {date}",
        f"appointments for {date}",
        f"my schedule {date}",
        f"calander {date}",
        f"{date} को कैलेंडर",  # Hindi
        f"calender {date}",
        f"apointments {date}",
        f"shedule for {date}",
        f"events on {date} please",
        f"what do i have on {date}",
        f"any meetings {date}",
        f"check my day {date}",
        f"{date} کا شیڈیول دکھائیں",  # Urdu
        f"{date} کیلنڈر دیکھنا ہے",  # Urdu
    ]
    return random.choice(templates)

def generate_calendar_create_prompts(date, title):
    templates = [
        f"Add event {title} on {date}",
        f"Create calendar entry {title} for {date}",
        f"Schedule {title} on {date}",
        f"book meeting {title} {date}",
        f"add to calender {title} {date}",
        f"Create an appointment: {title} on {date}",
        f"I need to schedule {title} for {date}",
        f"Put {title} on my calendar for {date}",
        f"set up {title} {date}",
    ]
    return random.choice(templates)

def generate_convert_prompts(value, from_unit, to_unit):
    templates = [
        f"Convert {value} {from_unit} to {to_unit}",
        f"{value} {from_unit} in {to_unit}",
        f"How many {to_unit} is {value} {from_unit}?",
        f"{value} {from_unit} -> {to_unit}",
        f"{value} {from_unit} equals how many {to_unit}",
        f"convert {value} {from_unit} to {to_unit}",
        f"{value} {from_unit} to {to_unit}",
        f"conver {value} {from_unit} to {to_unit}",
        f"covnert {value} {from_unit} {to_unit}",
        f"{value} {from_unit} ino {to_unit}",
        f"change {value} {from_unit} to {to_unit}",
        f"कनवर्ट {value} {from_unit} to {to_unit}",
        f"What is {value} {from_unit} in {to_unit}?",
        f"Can you calculate {value} {from_unit} to {to_unit}?",
        f"Help me convert {value} {from_unit} please",
        f"{value} {from_unit} - how much is that in {to_unit}?",
    ]
    return random.choice(templates)

def generate_currency_prompts(amount, from_iso3, to_iso3):
    templates = [
        f"Convert {amount} {from_iso3} to {to_iso3}",
        f"{amount} {from_iso3} in {to_iso3}",
        f"How many {to_iso3} is {amount} {from_iso3}?",
        f"{amount} {from_iso3} -> {to_iso3}",
        f"{amount} dollars to euros",
        f"{amount} usd to gbp",
        f"exchange {amount} {from_iso3} too {to_iso3}",
        f"currancy convert {amount} {from_iso3}",
        f"{amount} {from_iso3} -> {to_iso3} plz",
        f"मुद्रा {amount} {from_iso3} से {to_iso3}",  # Hindi
        f"convierte {amount} {from_iso3} a {to_iso3}",  # Spanish
        f"کرنسی {amount} {from_iso3} سے {to_iso3}",  # Urdu
        f"پیسہ {amount} {from_iso3} {to_iso3} میں بدلیں",  # Urdu
        f"How much is {amount} {from_iso3} worth in {to_iso3}?",
        f"What do I get for {amount} {from_iso3} in {to_iso3}?",
        f"Exchange {amount} {from_iso3} to {to_iso3}",
        f"Convert my {amount} {from_iso3} please",
    ]
    return random.choice(templates)

def generate_data():
    data = []
    id_counter = 0

    # STANDARD TOOL CALLS - 20%
    for _ in range(150):
        location = random.choice(LOCATIONS)
        prompt = generate_weather_prompts(location)
        data.append(create_example(prompt, "weather", {"location": location, "unit": random.choice(UNITS_C)}, False))
        id_counter += 1

    for _ in range(75):
        date = generate_date()
        prompt = generate_calendar_list_prompts(date)
        data.append(create_example(prompt, "calendar", {"action": "list", "date": date}, False))
        id_counter += 1

    for _ in range(75):
        date = generate_date()
        title = random.choice(["Meeting", "Dentist", "Gym", "Call", "Lunch", "Workshop", "Interview", "Doctor", "Flight", "Party", "Class", "Study", "Yoga", "Massage", "Haircut"])
        prompt = generate_calendar_create_prompts(date, title)
        data.append(create_example(prompt, "calendar", {"action": "create", "date": date, "title": title}, False))
        id_counter += 1

    for _ in range(100):
        value = random.randint(1, 10000) / random.choice([1, 10, 100])
        from_unit = random.choice(UNITS_CONVERT)
        to_unit = random.choice(UNITS_CONVERT)
        while from_unit == to_unit:
            to_unit = random.choice(UNITS_CONVERT)
        prompt = generate_convert_prompts(value, from_unit, to_unit)
        data.append(create_example(prompt, "convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit}, False))
        id_counter += 1

    for _ in range(100):
        amount = random.randint(1, 50000)
        from_iso3 = random.choice(ISO_CODES)
        to_iso3 = random.choice(ISO_CODES)
        while from_iso3 == to_iso3:
            to_iso3 = random.choice(ISO_CODES)
        prompt = generate_currency_prompts(amount, from_iso3, to_iso3)
        data.append(create_example(prompt, "currency", {"amount": amount, "from": from_iso3, "to": to_iso3}, False))
        id_counter += 1

    for _ in range(40):
        query = random.choice(["SELECT * FROM users", "SELECT * FROM employees", "SELECT name, email FROM customers", "SELECT * FROM products", "SELECT count(*) FROM orders", "SELECT * FROM users WHERE active = true", "SELECT * FROM events ORDER BY date"])
        data.append(create_example(query, "sql", {"query": query}, False))
        id_counter += 1

    # PARAPHRASED - 15%
    for _ in range(100):
        location = random.choice(LOCATIONS)
        variations = [
            f"Give me the weather report for {location}",
            f"I need to know the weather in {location}",
            f"Check the forecast for {location}",
            f"How's the weather looking in {location} today",
            f"What's the current temperature in {location}?",
            f"please tell me the weather {location}",
            f"looking for weather in {location}",
        ]
        prompt = random.choice(variations)
        data.append(create_example(prompt, "weather", {"location": location, "unit": random.choice(UNITS_C)}, False))
        id_counter += 1

    for _ in range(50):
        value = random.randint(1, 10000) / random.choice([1, 10, 100])
        from_unit = random.choice(UNITS_CONVERT)
        to_unit = random.choice(UNITS_CONVERT)
        while from_unit == to_unit:
            to_unit = random.choice(UNITS_CONVERT)
        variations = [
            f"What is {value} {from_unit} in {to_unit}?",
            f"Can you calculate {value} {from_unit} to {to_unit}?",
            f"Help me convert {value} {from_unit} please",
            f"{value} {from_unit} - how much is that in {to_unit}?",
            f"I need to know {value} {from_unit} in {to_unit}",
        ]
        prompt = random.choice(variations)
        data.append(create_example(prompt, "convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit}, False))
        id_counter += 1

    for _ in range(50):
        amount = random.randint(1, 50000)
        from_iso3 = random.choice(ISO_CODES)
        to_iso3 = random.choice(ISO_CODES)
        while from_iso3 == to_iso3:
            to_iso3 = random.choice(ISO_CODES)
        variations = [
            f"How much is {amount} {from_iso3} worth in {to_iso3}?",
            f"What do I get for {amount} {from_iso3} in {to_iso3}?",
            f"Exchange {amount} {from_iso3} to {to_iso3}",
            f"Convert my {amount} {from_iso3} please",
            f"I have {amount} {from_iso3} - what is that in {to_iso3}?",
        ]
        prompt = random.choice(variations)
        data.append(create_example(prompt, "currency", {"amount": amount, "from": from_iso3, "to": to_iso3}, False))
        id_counter += 1

    # ADVERSARIAL - 35%
    for _ in range(150):
        location = random.choice(LOCATIONS)
        variations = [
            f"wheather in {location}",
            f"weater {location}",
            f"weather in {location} pleez",
            f"wthr report {location}",
            f"weathr forcast {location}",
            f"मौसम {location}",
            f"clima de {location}",
            f"الطقس {location}",
            f"weathar {location}",
            f"wetherr {location}",
            f"wheather in {location}",
            f"weater report {location}",
            f"wthr {location}",
            f"weather forcast {location}",
            f"मौसम {location} क्या है",
            f"clima del {location}",
            f"climate {location}",
            f"तापमान {location}",
        ]
        prompt = random.choice(variations)
        data.append(create_example(prompt, "weather", {"location": location, "unit": random.choice(UNITS_C)}, False))
        id_counter += 1

    for _ in range(80):
        value = random.randint(1, 10000) / random.choice([1, 10, 100])
        from_unit = random.choice(UNITS_CONVERT)
        to_unit = random.choice(UNITS_CONVERT)
        while from_unit == to_unit:
            to_unit = random.choice(UNITS_CONVERT)
        variations = [
            f"conver {value} {from_unit} to {to_unit}",
            f"covnert {value} {from_unit} {to_unit}",
            f"{value} {from_unit} ino {to_unit}",
            f"change {value} {from_unit} to {to_unit}",
            f"कनवर्ट {value} {from_unit} to {to_unit}",
            f"convers {value} {from_unit} to {to_unit}",
            f"cnvrt {value} {from_unit} {to_unit}",
            f"क्या {value} {from_unit} {to_unit} मे बदल सकते हो",
        ]
        prompt = random.choice(variations)
        data.append(create_example(prompt, "convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit}, False))
        id_counter += 1

    for _ in range(80):
        amount = random.randint(1, 50000)
        from_iso3 = random.choice(ISO_CODES)
        to_iso3 = random.choice(ISO_CODES)
        while from_iso3 == to_iso3:
            to_iso3 = random.choice(ISO_CODES)
        variations = [
            f"exchange {amount} {from_iso3} too {to_iso3}",
            f"currancy convert {amount} {from_iso3}",
            f"{amount} {from_iso3} -> {to_iso3} plz",
            f"मुद्रा {amount} {from_iso3} से {to_iso3}",
            f"convierte {amount} {from_iso3} a {to_iso3}",
            f"exchng {amount} {from_iso3} to {to_iso3}",
            f"currncy {amount} {from_iso3} {to_iso3}",
            f"क्या {amount} {from_iso3} {to_iso3} मे बदल सकते हो",
        ]
        prompt = random.choice(variations)
        data.append(create_example(prompt, "currency", {"amount": amount, "from": from_iso3, "to": to_iso3}, False))
        id_counter += 1

    for _ in range(50):
        date = generate_date()
        variations = [
            f"calander {date}",
            f"apointments {date}",
            f"shedule for {date}",
            f"कार्यक्रम {date}",
            f"calendario {date}",
            f"apointmnt {date}",
            f"schdule {date}",
        ]
        prompt = random.choice(variations)
        data.append(create_example(prompt, "calendar", {"action": "list", "date": date}, False))
        id_counter += 1

    # REFUSALS - 30%
    refusal_prompts = [
        "Hello, how are you?",
        "What's your name?",
        "Tell me a joke",
        "Who are you?",
        "What is 2+2?",
        "What is the meaning of life?",
        "Can you dance?",
        "Write me a poem",
        "What do you think about AI?",
        "Hello there!",
        "Hi assistant",
        "How's your day going?",
        "Tell me about yourself",
        "What's the best movie?",
        "I love coding!",
        "Can you help me with my homework?",
        "What's up?",
        "Good morning!",
        "Thanks for nothing",
        "Are you real?",
        "Make me a sandwich",
        "What is the weather like today?",
        "Convert currency",
        "Schedule something",
        "I want to fly to the moon",
        "Can you read my mind?",
        "What should I eat for dinner?",
        "Tell me a secret",
        "What's the best programming language?",
        "Do you like pizza?",
        "Sing me a song",
        "What is love?",
        "Tell me something interesting",
        "What's your favorite color?",
        "Can you beatbox?",
        "What's the stock price of Apple?",
        "Who will win the Super Bowl?",
        "What's the best restaurant in town?",
        "I want to learn Spanish",
        "What's 1+1=?",
        "Calculate infinity plus one",
        "Find the meaning of 42",
        "What happens after death?",
        "Is there life on Mars?",
        "Can you predict the future?",
        "What's the best vacation spot?",
        "Tell me about quantum physics",
        "What is consciousness?",
        "Explain the universe",
    ]
    # Generate 30% refusals
    target_refusals = int(len(data) * 0.30)
    for _ in range(target_refusals):
        prompt = random.choice(refusal_prompts)
        data.append(create_example(prompt, None, None, True))
        id_counter += 1

    # MULTI-TURN EXAMPLES - with explicit history
    multi_turn_examples = [
        # Turn 1: weather, Turn 2: convert temperature
        ({"prompt": "What's the weather in Paris?", "tool": "weather", "args": {"location": "Paris", "unit": "C"}},
         {"prompt": "Now convert that temperature to fahrenheit", "tool": "convert", "args": None}),
        
        ({"prompt": "weather in London", "tool": "weather", "args": {"location": "London", "unit": "C"}},
         {"prompt": "convert that to F", "tool": "convert", "args": None}),
        
        ({"prompt": "temperature in Tokyo", "tool": "weather", "args": {"location": "Tokyo", "unit": "C"}},
         {"prompt": "what's that in fahrenheit", "tool": "convert", "args": None}),
         
        # Turn 1: currency, Turn 2: convert to another
        ({"prompt": "Convert 100 USD to EUR", "tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}},
         {"prompt": "What about in GBP?", "tool": "currency", "args": None}),
        
        ({"prompt": "50 euros to dollars", "tool": "currency", "args": {"amount": 50, "from": "EUR", "to": "USD"}},
         {"prompt": "and in yen?", "tool": "currency", "args": None}),
        
        ({"prompt": "exchange 200 GBP to USD", "tool": "currency", "args": {"amount": 200, "from": "GBP", "to": "USD"}},
         {"prompt": "how about euros", "tool": "currency", "args": None}),
        
        # Turn 1: calendar list, Turn 2: create event
        ({"prompt": "Show my calendar for 2026-05-15", "tool": "calendar", "args": {"action": "list", "date": "2026-05-15"}},
         {"prompt": "Add a meeting called Team Standup on that day", "tool": "calendar", "args": None}),
        
        ({"prompt": "what's on my schedule tomorrow", "tool": "calendar", "args": None},
         {"prompt": "add dentist appointment", "tool": "calendar", "args": None}),
         
        # Weather chain
        ({"prompt": "Is it going to rain in Sydney?", "tool": "weather", "args": {"location": "Sydney", "unit": "C"}},
         {"prompt": "what about in Melbourne", "tool": "weather", "args": None}),
         
        # Convert chain
        ({"prompt": "how many meters in 5 feet?", "tool": "convert", "args": {"value": 5, "from_unit": "feet", "to_unit": "meters"}},
         {"prompt": "and in centimeters?", "tool": "convert", "args": None}),
    ]
    
    # Add multi-turn with history
    for turn1, turn2 in multi_turn_examples:
        # Calculate what the second turn args should be based on first turn
        if turn1["tool"] == "weather" and turn2["tool"] == "convert":
            temp = 20  # Assume default
            turn2_args = {"value": temp, "from_unit": "celsius", "to_unit": "fahrenheit"}
        elif turn1["tool"] == "currency" and turn2.get("args") is None:
            turn2_args = {"amount": turn1["args"]["amount"], "from": turn1["args"]["to"], "to": random.choice(ISO_CODES)}
        elif turn1["tool"] == "convert" and turn2.get("args") is None:
            turn2_args = {"value": turn1["args"]["value"], "from_unit": turn1["args"]["to_unit"], "to_unit": "centimeters"}
        else:
            turn2_args = turn2.get("args", {})
        
        # First turn with no history (treated as single turn)
        data.append(create_example(turn1["prompt"], turn1["tool"], turn1["args"], False))
        id_counter += 1
        
        # Second turn WITH history
        history_example = {
            "prompt": turn2["prompt"],
            "tool": turn2["tool"],
            "args": turn2_args,
            "is_refusal": False,
            "history": [{"role": "user", "content": turn1["prompt"]}]
        }
        data.append(history_example)
        id_counter += 1

    random.shuffle(data)
    for i, item in enumerate(data):
        item["id"] = i

    return data

if __name__ == "__main__":
    data = generate_data()
    print(f"Generated {len(data)} examples")

    test_size = 20
    test_data = data[:test_size]
    test_prompts = set(item["prompt"] for item in test_data)
    train_data = [item for item in data[test_size:] if item["prompt"] not in test_prompts]

    refusal_count = sum(1 for item in train_data if item["is_refusal"])
    print(f"Training: {len(train_data)} examples, Refusals: {refusal_count} ({100*refusal_count/len(train_data):.1f}%)")
    print(f"Test: {len(test_data)} examples")

    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open("data/test.jsonl", "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved to data/train.jsonl and data/test.jsonl")