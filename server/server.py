from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from server.database import save_combination, update_liked, count_combinations, get_all_combinations
from core.style_model import train_model, predict
from services.weather import get_weather

# FASTAPI SERVER
# client.py sends requests to this server.
# The server handles all database and ML operations.
# To run the server, type this in terminal:
#   uvicorn server:app --reload
# The server runs at: http://127.0.0.1:8000

# Create the FastAPI app
app = FastAPI(title="StyleMate API")

# When the server starts, we train the model right away
# So it's ready to make predictions immediately
print("Server starting — training ML model...")
current_model, current_scaler = train_model()

if current_model is None:
    print("Not enough data yet. Model will activate after 50+ combinations.")
else:
    print("Model ready!")

class SaveCombinationRequest(BaseModel):
    top:          str         
    bottom:       str         
    combo_vector: List[float] 


class LikeRequest(BaseModel):
    top:    str   
    bottom: str   

class PredictRequest(BaseModel):
    combo_vector: List[float]  


# ENDPOINTS
# Each endpoint is a function with a decorator like @app.get() or @app.post()
# GET  = client is ASKING for information (like asking a question)
# POST = client is SENDING data (like submitting a form)


@app.get("/")
def root():
    """
    Basic check — just confirms the server is running.
    Visit http://127.0.0.1:8000 in your browser to see this.
    """
    return {"message": "StyleMate API is running!"}


@app.get("/status")
def status():
    """
    Returns how many combinations are in the database.
    client.py uses this to show the DB counter on screen.
    """
    total, liked = count_combinations()
    return {
        "total":       total,
        "liked":       liked,
        "model_ready": current_model is not None
    }


@app.post("/save_combination")
def save_combination_endpoint(request: SaveCombinationRequest):
    """
    Saves a new combination to the database with liked=0.
    Called by client.py every time the user sees a new outfit combination.

    client.py sends:
      { "top": "outfit1.png", "bottom": "outfit_1.png",
        "combo_vector": [0.1, 2.6, 0.0, 94.1, 1.0, 104.2, 11.6, 108.4, 163.1, 0.0] }

    Returns:
      { "status": "saved" } or { "status": "already exists" }
    """
    save_combination(request.top, request.bottom, request.combo_vector)
    return {"status": "saved", "top": request.top, "bottom": request.bottom}


@app.post("/like")
def like_endpoint(request: LikeRequest):
    """
    Updates liked=1 for a combination AND retrains the ML model.
    Called by client.py when the user does a thumbs up gesture.

    client.py sends:
      { "top": "outfit1.png", "bottom": "outfit_1.png" }

    After updating the database, the model retrains so it learns
    from the new like immediately.

    Returns:
      { "status": "liked", "new_accuracy": 0.743 }
    """
    global current_model, current_scaler

    # Update the database
    update_liked(request.top, request.bottom)

    # Retrain the model with the new data
    print(f"Retraining model after like: {request.top} + {request.bottom}")
    current_model, current_scaler = train_model()

    # Calculate new accuracy to return to client
    if current_model is not None:
        from server.database import get_all_combinations
        import numpy as np
        all_data    = get_all_combinations()
        X           = [row["combo_vector"] for row in all_data if len(row["combo_vector"]) > 0]
        y           = [row["liked"] for row in all_data if len(row["combo_vector"]) > 0]
        X_scaled    = current_scaler.transform(np.array(X))
        new_accuracy = current_model.score(X_scaled, np.array(y))
        print(f"Model retrained! New accuracy: {new_accuracy:.1%}")
    else:
        new_accuracy = 0.0

    return {
        "status":       "liked",
        "top":          request.top,
        "bottom":       request.bottom,
        "new_accuracy": round(new_accuracy, 3)
    }


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    """
    Returns the match prediction percentage for a combo vector.
    Called by client.py every time the outfit changes.

    client.py sends:
      { "combo_vector": [0.1, 2.6, 0.0, 94.1, 1.0, 104.2, 11.6, 108.4, 163.1, 0.0] }

    Returns:
      { "match_score": 0.68, "match_percent": 68, "model_ready": true }
    """
    if current_model is None:
        # Model not trained yet — return neutral 50%
        return {
            "match_score":   0.5,
            "match_percent": 50,
            "model_ready":   False
        }

    score = predict(current_model, current_scaler, request.combo_vector)

    return {
        "match_score":   round(score, 3),
        "match_percent": int(score * 100),
        "model_ready":   True
    }

@app.get("/weather")
def weather_endpoint(city: str = "Istanbul"):
    """
    Fetches current weather for a city.
    client.py calls this every 10 minutes to update weather on screen.
    Returns weather data from OpenWeatherMap API.
    """
    result = get_weather(city)
    return result