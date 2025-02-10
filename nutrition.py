import requests
import json

API_KEY = "03c2db866375b500ec625e79aec2ce5d"
APP_ID = "e1711dde"

url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'

headers = {
  'x-app-id': APP_ID,
  'x-app-key': API_KEY,
  'Content-Type': 'application/json'
}

foods = ['Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Sandwich', 'Taco', 'Taquito', 'apple_pie', 'burger', 'butter_naan', 'chai', 'chapati', 'cheesecake', 'chicken_curry', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'ice_cream', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'omelette', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa', 'sushi']

def extract_nutrition_info(food_data):
  return {
    "calories": food_data.get('nf_calories'),
    "total_fat": food_data.get('nf_total_fat'),
    "carbohydrates": food_data.get('nf_total_carbohydrate'),
    "fiber": food_data.get('nf_dietary_fiber'),
    "sugars": food_data.get('nf_sugars'),
    "protein": food_data.get('nf_protein'),
    "serving_size": f"{food_data.get('serving_unit')}"
  }

nutrition_data = {}
for food in foods:
  response = requests.post(url, headers=headers, json={"query": food})

  if response.status_code == 200:
    data = response.json()

    if 'foods' in data and len(data['foods']) > 0:
      food_data = data['foods'][0]
      nutrition_data[food] = extract_nutrition_info(food_data)
    else:
      print(f"Warning: No data found for {food}")
  else:
    print(f"Error fetching data for {food}: {response.status_code}")

# Save to a JSON file
with open('food_nutrition.json', 'w') as f:
  json.dump(nutrition_data, f, indent=4)

print("Nutrition data saved to 'food_nutrition.json'")

