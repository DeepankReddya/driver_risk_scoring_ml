def calculate_risk_score(probability):

    score = int(probability * 100)

    if score < 33:
        category = "Safe Driver"

    elif score < 66:
        category = "Moderate Risk"

    else:
        category = "High Risk"

    return score, category