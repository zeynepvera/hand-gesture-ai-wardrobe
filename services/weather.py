import requests

# WEATHER MODULE
# This file has one job: fetch current weather data from OpenWeatherMap API
# and return it in a simple format that the rest of the app can use.

API_KEY = "431b0b7831e49f4494476c63ac7180c2"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

DEFAULT_CITY = "Istanbul"

def get_my_city():
    """
    Automatically detects the user's city using their IP address.
    HOW IT WORKS:
      We call a free service called ip-api.com
      It looks at our IP address and guesses our location
      No API key needed — completely free
    """
    try:
        print("Detecting your location from IP address...")
 
        # ip-api.com is a free service — no API key needed
        response = requests.get("http://ip-api.com/json/", timeout=5)
 
        if response.status_code == 200:
            data = response.json()
 
            if data.get("status") == "success":
                city    = data.get("city", DEFAULT_CITY)
                country = data.get("country", "")
                print(f"Location detected: {city}, {country}")
                return city
            else:
                print(f"IP detection failed: {data.get('message', 'unknown error')}")
                print(f"Using default city: {DEFAULT_CITY}")
                return DEFAULT_CITY
        else:
            print(f"IP detection failed: HTTP {response.status_code}")
            print(f"Using default city: {DEFAULT_CITY}")
            return DEFAULT_CITY
 
    except requests.exceptions.Timeout:
        print("IP detection timed out.")
        print(f"Using default city: {DEFAULT_CITY}")
        return DEFAULT_CITY
 
    except requests.exceptions.ConnectionError:
        print("No internet connection for IP detection.")
        print(f"Using default city: {DEFAULT_CITY}")
        return DEFAULT_CITY
 
    except Exception as e:
        print(f"IP detection error: {e}")
        print(f"Using default city: {DEFAULT_CITY}")
        return DEFAULT_CITY

def clean_turkish(text):
    # OpenCV font doesn't support Turkish letters, replace with English
    replacements = {
        'ı': 'i', 'İ': 'I',
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ş': 's', 'Ş': 'S',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C'
    }
    for turkish, english in replacements.items():
        text = text.replace(turkish, english)
    return text

def get_weather(city):
    """
    Fetches current weather for a given city.

    HOW IT WORKS:
      1. We send a GET request to OpenWeatherMap API with the city name
      2. The API returns a JSON response with weather data
      3. We extract the parts we need (temp, condition)
      4. We return a simple dictionary
    """

    # Build the request 
    # We send these parameters with our request:

    params = {
        "q":     city,
        "appid": API_KEY,
        "units": "metric" #celsius
    }
 
    try:
        response = requests.get(BASE_URL, params=params, timeout=5)
 
        if response.status_code == 401:
            return {"success": False, "error": "Invalid API key"}
        if response.status_code == 404:
            return {"success": False, "error": f"City '{city}' not found"}
        if response.status_code != 200:
            return {"success": False, "error": f"API error: {response.status_code}"}
 
        data = response.json()
 
        temp       = round(data["main"]["temp"])
        feels_like = round(data["main"]["feels_like"])
        humidity   = data["main"]["humidity"]
        condition  = data["weather"][0]["main"]
        city_clean = clean_turkish(data["name"])  
 
        return {
            "success":    True,
            "city":       city_clean,
            "temp":       temp,
            "feels_like": feels_like,
            "humidity":   humidity,
            "condition":  condition,
        }
 
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Weather request timed out"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "No internet connection"}
    except Exception as e:
        return {"success": False, "error": str(e)}

