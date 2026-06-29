import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from server.database import get_all_combinations, count_combinations

# STYLE MODEL
# This file has one job: learn from your liked combinations and predict
# whether you will like a new combination or not.
# It uses Logistic Regression — the simplest ML model for yes/no questions.


# Minimum number of combinations needed before we can train the model
MIN_COMBINATIONS = 50


def train_model():
    """
    The problem: mean_brightness (0-255) has much bigger numbers than
    warm_cool_ratio (0-1). The model might think brightness is more
    important just because the numbers are bigger — but that's wrong!
    This process is called "feature normalization" or "standardization".
    """

    total, liked = count_combinations()

    if total < MIN_COMBINATIONS:
        print(f"Not enough data to train. Have {total}, need {MIN_COMBINATIONS}.")
        return None, None

    if liked < 5:
   
        print(f"Not enough liked combinations. Have {liked}, need at least 5.")
        return None, None

    print(f"Training model with {total} combinations ({liked} liked)...")

    all_data = get_all_combinations()

    # X = list of combo vectors (each is a list of 10 numbers)
    # y = list of liked values  (each is 0 or 1)
    X = []
    y = []

    for row in all_data:
        if len(row["combo_vector"]) == 0:
            continue
        X.append(row["combo_vector"])
        y.append(row["liked"])

    # Convert to numpy arrays — scikit-learn works with numpy arrays
    X = np.array(X)
    y = np.array(y)


    # This makes all features equally important to the model
    scaler = StandardScaler()

    # fit_transform does two things:
    #   fit      → learns the mean and std of each feature from our data
    #   transform → applies the scaling to our data
    X_scaled = scaler.fit_transform(X)

    # Train the model
    # max_iter=1000 means the model tries up to 1000 times to find the best fit
    # C=1.0 controls how flexible the model is
    #   Low C  → model is simple, might underfit (miss patterns)
    #   High C → model is complex, might overfit (memorize instead of learn)
    #   C=1.0  → good default balance
    model = LogisticRegression(max_iter=1000, C=1.0)

    # model.fit() looks at all the X vectors and y labels
    # and figures out the pattern between them
    model.fit(X_scaled, y)

    print(f"Model trained successfully!")

    
    accuracy = model.score(X_scaled, y)
    print(f"Training accuracy: {accuracy:.1%}")
    # Note: high accuracy on training data doesn't always mean the model
    # is good — it might just be memorizing. But for our small dataset it's fine.

    return model, scaler


def predict(model, scaler, combo_vector):
    
    if model is None or scaler is None:
        return 0.5

    # Convert combo_vector to a 2D numpy array
    # scikit-learn expects shape (1, 10) not (10,)
    # reshape(1, -1) means: "1 row, figure out the columns automatically"
    X = np.array(combo_vector).reshape(1, -1)

    # Scale the new combo vector using the SAME scaler we trained with
    # Important: we use transform() not fit_transform() here
    # Because we don't want to relearn the scaling — just apply it
    X_scaled = scaler.transform(X)

    # predict_proba returns two values: [probability of 0, probability of 1]
    # Example: [[0.13, 0.87]] means 13% chance of not liking, 87% chance of liking
    # We take index [0][1] to get the "liked" probability
    probability = model.predict_proba(X_scaled)[0][1]

    return float(probability)

