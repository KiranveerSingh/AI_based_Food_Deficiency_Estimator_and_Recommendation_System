import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Load Models & Data ---
MODEL_PATH = os.path.join('Models', 'finalbest.pkl')
ENCODER_PATH = os.path.join('Models', 'deficiency_feature_labelencoder.pkl')
FOOD_DATA_PATH = os.path.join('Dataset', 'cleaned_food_nutrition_dataset.csv')  # Adjust path

# Load prediction model
model = joblib.load(MODEL_PATH)

# Load label encoder if available
if os.path.exists(ENCODER_PATH):
    le = joblib.load(ENCODER_PATH)
    encoder_available = True
else:
    encoder_available = False

# Load food nutrition dataset
food_df = pd.read_csv(FOOD_DATA_PATH).fillna(0)

# Deficiency to nutrient column mapping (adjust names to match your dataset)
DEFICIENCY_COLUMN_MAP = {
    'Iron': 'Iron_Intake_(mg)',
    'Calcium': 'Calcium_Intake_(mg)',
    'Vitamin B12': 'Vitamin_B12_Intake_(mcg)',
    'Vitamin D': 'Vitamin_D_Intake_(IU)',
    'Folate': 'Folate_(Folic_Acid)_Intake_(mcg)',
    'Vitamin C': 'Vitamin_C_Intake_(mg)',
    'Zinc': 'Zinc_Intake_(mg)',
    'Magnesium': 'Magnesium_Intake_(mg)',
    # Add other nutrients as needed
}

# --- Streamlit UI ---

st.set_page_config(page_title="Nutrition Deficiency & Food Recommendations", layout="wide")


st.title("🍎 Nutrition Deficiency Predictor & Food Recommendation")
st.markdown("Enter your details, predict nutritional deficiencies, and get personalized food recommendations.")

# Sidebar Filters
st.sidebar.header("Customize Food Recommendations")
user_allergens = st.sidebar.multiselect(
    "Exclude Allergens:",
    options=['nuts', 'gluten', 'dairy', 'soy', 'eggs', 'shellfish']
)
user_diet_tags = st.sidebar.multiselect(
    "Include Only Diets:",
    options=['vegetarian', 'vegan', 'keto', 'paleo', 'gluten-free', 'low-carb']
)

# User Inputs for prediction (outside form for live BMI and calorie calculation)
Age = st.number_input("Age", 1, 100, value=25)
Height_cm = st.number_input("Height (cm)", 50, 350, value=170)
Weight_kg = st.number_input("Weight (kg)", 20, 300, value=70)

height_m = Height_cm / 100 if Height_cm > 0 else 1
BMI = Weight_kg / (height_m ** 2) if height_m > 0 else 0
st.markdown(f"**Calculated BMI:** {BMI:.2f}")

Daily_Protein_Intake_g = st.number_input("Daily Protein Intake (g)", 0, 300, value=60)
Daily_Fat_Intake_g = st.number_input("Daily Fat Intake (g)", 0, 500, value=70)
Daily_Carbohydrate_Intake_g = st.number_input("Daily Carbohydrate Intake (g)", 0, 1000, value=250)

Daily_Calorie_Intake = (
    Daily_Protein_Intake_g * 4 +
    Daily_Carbohydrate_Intake_g * 4 +
    Daily_Fat_Intake_g * 9
)
st.markdown(f"**Calculated Daily Calorie Intake:** {Daily_Calorie_Intake:.0f} kcal")

# Other inputs inside form for clarity
with st.form("deficiency_form"):
    Daily_Fiber_Intake_g = st.number_input("Daily Fiber Intake (g)", 0, 150, value=25)
    Daily_Water_Intake_liters = st.number_input("Daily Water Intake (liters)", 0.0, 15.0, value=2.0, step=0.1)
    Vitamin_A_Intake_mcg = st.number_input("Vitamin A Intake (mcg)", 0, 5000, value=900)
    Vitamin_B1_Intake_mg = st.number_input("Vitamin B1 (Thiamine) Intake (mg)", 0.0, 50.0, value=1.2, step=0.01)
    Vitamin_B2_Intake_mg = st.number_input("Vitamin B2 (Riboflavin) Intake (mg)", 0.0, 50.0, value=1.3, step=0.01)
    Vitamin_B3_Intake_mg = st.number_input("Vitamin B3 (Niacin) Intake (mg)", 0.0, 100.0, value=16.0, step=0.01)
    Vitamin_B5_Intake_mg = st.number_input("Vitamin B5 (Pantothenic Acid) Intake (mg)", 0.0, 100.0, value=5.0, step=0.01)
    Vitamin_B6_Intake_mg = st.number_input("Vitamin B6 Intake (mg)", 0.0, 50.0, value=1.3, step=0.01)
    Vitamin_B12_Intake_mcg = st.number_input("Vitamin B12 Intake (mcg)", 0.0, 50.0, value=2.4, step=0.01)
    Vitamin_C_Intake_mg = st.number_input("Vitamin C Intake (mg)", 0.0, 500.0, value=90.0, step=0.1)
    Vitamin_D_Intake_IU = st.number_input("Vitamin D Intake (IU)", 0, 5000, value=600)
    Vitamin_E_Intake_mg = st.number_input("Vitamin E Intake (mg)", 0.0, 100.0, value=15.0, step=0.1)
    Vitamin_K_Intake_mcg = st.number_input("Vitamin K Intake (mcg)", 0, 2000, value=120)
    Iron_Intake_mg = st.number_input("Iron Intake (mg)", 0.0, 100.0, value=15.0, step=0.1)
    Zinc_Intake_mg = st.number_input("Zinc Intake (mg)", 0.0, 100.0, value=15.0, step=0.1)
    Magnesium_Intake_mg = st.number_input("Magnesium Intake (mg)", 0, 1000, value=400)
    Calcium_Intake_mg = st.number_input("Calcium Intake (mg)", 0, 3000, value=1000)
    Iodine_Intake_mcg = st.number_input("Iodine Intake (mcg)", 0, 1000, value=150)
    Selenium_Intake_mcg = st.number_input("Selenium Intake (mcg)", 0, 500, value=55)
    Copper_Intake_mg = st.number_input("Copper Intake (mg)", 0.0, 10.0, value=0.9, step=0.01)
    Folate_Intake_mcg = st.number_input("Folate (Folic Acid) Intake (mcg)", 0, 2000, value=400)
    Phosphorus_Intake_mg = st.number_input("Phosphorus Intake (mg)", 0, 2000, value=700)
    Manganese_Intake_mg = st.number_input("Manganese Intake (mg)", 0.0, 10.0, value=2.3, step=0.01)
    Chromium_Intake_mcg = st.number_input("Chromium Intake (mcg)", 0, 1000, value=35)
    Food_Diversity_Score = st.number_input("Food Diversity Score", 0, 100, value=50)
    Hemoglobin_g_dl = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, value=14.0, step=0.1)
    Serum_Ferritin_ng_ml = st.number_input("Serum Ferritin (ng/mL)", 0.0, 500.0, value=100.0, step=0.1)
    Vitamin_D_25_OH_ng_ml = st.number_input("Vitamin D 25-OH (ng/mL)", 0.0, 100.0, value=30.0, step=0.1)
    Serum_B12_pg_ml = st.number_input("Serum B12 (pg/mL)", 0.0, 2000.0, value=400.0, step=0.1)
    Serum_Folate_ng_ml = st.number_input("Serum Folate (ng/mL)", 0.0, 25.0, value=10.0, step=0.1)
    C_Reactive_Protein_mg_dl = st.number_input("C-Reactive Protein (mg/dL)", 0.0, 10.0, value=1.0, step=0.01)
    Serum_Albumin_g_dl = st.number_input("Serum Albumin (g/dL)", 0.0, 10.0, value=4.5, step=0.01)
    Total_Cholesterol_mg_dl = st.number_input("Total Cholesterol (mg/dL)", 0, 400, value=200)
    Fasting_Glucose_mg_dl = st.number_input("Fasting Glucose (mg/dL)", 0, 400, value=90)
    Alcohol_Consumption_g_day = st.number_input("Alcohol Consumption (g/day)", 0.0, 200.0, value=0.0, step=0.1)
    Medication_Interaction_Score = st.number_input("Medication Interaction Score", 0, 100, value=0)
    
    submitted = st.form_submit_button("Predict Deficiency")

def recommend_foods(food_df, deficiency, top_n=5, exclude_allergens=None, include_tags=None):
    nutrient_col = DEFICIENCY_COLUMN_MAP.get(deficiency)
    if nutrient_col is None or nutrient_col not in food_df.columns:
        st.warning(f"No nutrient data available for '{deficiency}'.")
        return pd.DataFrame()
    foods = food_df[food_df[nutrient_col] > 0].copy()
    if exclude_allergens:
        for allergen in exclude_allergens:
            foods = foods[~foods['Allergens'].str.contains(allergen, case=False, na=False)]
    if include_tags:
        include_mask = pd.Series(False, index=foods.index)
        for tag in include_tags:
            include_mask = include_mask | foods['Tags'].str.contains(tag, case=False, na=False)
        foods = foods[include_mask]
    foods = foods.sort_values(nutrient_col, ascending=False).reset_index(drop=True)
    foods.insert(0, 'Rank', range(1, len(foods) + 1))
    cols = ['Rank', 'Food_Item', nutrient_col, 'Calories', 'Health_Benefits']
    cols = [c for c in cols if c in foods.columns]
    recommended = foods[cols].head(top_n)
    recommended.rename(columns={nutrient_col: f"{deficiency} per 100g"}, inplace=True)
    return recommended

if submitted:
    user_data = pd.DataFrame([{
        "Age": Age,
        "Height_(cm)": Height_cm,
        "Weight_(kg)": Weight_kg,
        "BMI": BMI,
        "Daily_Calorie_Intake": Daily_Calorie_Intake,
        "Daily_Protein_Intake_(g)": Daily_Protein_Intake_g,
        "Daily_Fat_Intake_(g)": Daily_Fat_Intake_g,
        "Daily_Carbohydrate_Intake_(g)": Daily_Carbohydrate_Intake_g,
        "Daily_Fiber_Intake_(g)": Daily_Fiber_Intake_g,
        "Daily_Water_Intake_(liters)": Daily_Water_Intake_liters,
        "Vitamin_A_Intake_(mcg)": Vitamin_A_Intake_mcg,
        "Vitamin_B1_(Thiamine)_Intake_(mg)": Vitamin_B1_Intake_mg,
        "Vitamin_B2_(Riboflavin)_Intake_(mg)": Vitamin_B2_Intake_mg,
        "Vitamin_B3_(Niacin)_Intake_(mg)": Vitamin_B3_Intake_mg,
        "Vitamin_B5_(Pantothenic_Acid)_Intake_(mg)": Vitamin_B5_Intake_mg,
        "Vitamin_B6_Intake_(mg)": Vitamin_B6_Intake_mg,
        "Vitamin_B12_Intake_(mcg)": Vitamin_B12_Intake_mcg,
        "Vitamin_C_Intake_(mg)": Vitamin_C_Intake_mg,
        "Vitamin_D_Intake_(IU)": Vitamin_D_Intake_IU,
        "Vitamin_E_Intake_(mg)": Vitamin_E_Intake_mg,
        "Vitamin_K_Intake_(mcg)": Vitamin_K_Intake_mcg,
        "Iron_Intake_(mg)": Iron_Intake_mg,
        "Zinc_Intake_(mg)": Zinc_Intake_mg,
        "Magnesium_Intake_(mg)": Magnesium_Intake_mg,
        "Calcium_Intake_(mg)": Calcium_Intake_mg,
        "Iodine_Intake_(mcg)": Iodine_Intake_mcg,
        "Selenium_Intake_(mcg)": Selenium_Intake_mcg,
        "Copper_Intake_(mg)": Copper_Intake_mg,
        "Folate_(Folic_Acid)_Intake_(mcg)": Folate_Intake_mcg,
        "Phosphorus_Intake_(mg)": Phosphorus_Intake_mg,
        "Manganese_Intake_(mg)": Manganese_Intake_mg,
        "Chromium_Intake_(mcg)": Chromium_Intake_mcg,
        "Food_Diversity_Score": Food_Diversity_Score,
        "Hemoglobin_(g/dL)": Hemoglobin_g_dl,
        "Serum_Ferritin_(ng/mL)": Serum_Ferritin_ng_ml,
        "Vitamin_D_25-OH_(ng/mL)": Vitamin_D_25_OH_ng_ml,
        "Serum_B12_(pg/mL)": Serum_B12_pg_ml,
        "Serum_Folate_(ng/mL)": Serum_Folate_ng_ml,
        "C-Reactive_Protein_(mg/dL)": C_Reactive_Protein_mg_dl,
        "Serum_Albumin_(g/dL)": Serum_Albumin_g_dl,
        "Total_Cholesterol_(mg/dL)": Total_Cholesterol_mg_dl,
        "Fasting_Glucose_(mg/dL)": Fasting_Glucose_mg_dl,
        "Alcohol_Consumption_(g/day)": Alcohol_Consumption_g_day,
        "Medication_Interaction_Score": Medication_Interaction_Score,
    }])

    preds = model.predict(user_data)

    st.subheader("Prediction Results")
    if preds.shape[-1] == 2:
        deficiency_code = int(preds[0][0])
        deficiency_count = int(preds[0][1])
    else:
        deficiency_code = int(preds[0])
        deficiency_count = None

    if encoder_available:
        predicted_deficiency = le.inverse_transform([deficiency_code])[0]
    else:
        predicted_deficiency = str(deficiency_code)

    st.markdown(f"**Predicted Deficiency:** {predicted_deficiency}")
    st.markdown(f"**Deficiency Count:** {deficiency_count}")

    # Show recommendations with filters as selected by user
    recommended_foods = recommend_foods(
        food_df,
        predicted_deficiency,
        top_n=5,
        exclude_allergens=user_allergens,
        include_tags=user_diet_tags
    )
    if not recommended_foods.empty:
        st.subheader(f"Recommended Foods to Overcome {predicted_deficiency} Deficiency")
        st.table(recommended_foods)
    else:
        st.warning("No suitable foods found with current filters.")
