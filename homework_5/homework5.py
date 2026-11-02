#scikit-learn==1.7.2, name = "scikit-learn", specifier = "==1.7.2" 
#uv 0.13.5 (x86_64-unknown-linux-gnu)
#Exact probability of conversion: 0.533
#{"conversion_probability":0.769799,"conversion":true}
#python:3.11.15-slim-bookworm
#{'conversion_probability': 0.769799, 'conversion': True}
#{"status":"ok","model_sha256":"1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1","feature_defaults_sha256":"1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"}

import pickle

# Define the lead data
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

# Load the pipeline artifact
with open('pipeline.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Predict probability [1] represents the probability of class 1 (conversion)
probs = pipeline.predict_proba([lead])[0]
prob_conversion = probs[1]

# Report the final exact score
print(f"Exact probability of conversion: {prob_conversion:.3f}")
