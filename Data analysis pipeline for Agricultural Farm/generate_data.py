import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

# Set random seed for generating same data each time 
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_PRODUCTIONS = 15000
NUM_SALES = 7500  # allow multiple sales per production

# Sample values for realism
crop_types = ['Wheat', 'Corn', 'Barley', 'Soybean', 'Potato']
varieties = {'Wheat': ['Durum', 'Emmer'], 'Corn': ['Sweet', 'Dent'], 'Barley': ['Hulled', 'Hulless'],
             'Soybean': ['Yellow', 'Black'], 'Potato': ['Russet', 'Red']}

fertilizers = ['NPK 15-15-15', 'Urea', 'Compost']
pesticides = ['Glyphosate', 'Chlorpyrifos', 'Neem Oil']
irrigation_types = ['Drip', 'Sprinkler', 'Flood']

buyer_types = ['Retailer', 'Wholesaler', 'Exporter']
channel_types = ['Direct', 'Online', 'Cooperative']

regions = ['Bavaria', 'Brandenburg', 'Saxony', 'Hesse', 'Lower Saxony']

def generate_production_data():
    records = []
    for i in range(1, NUM_PRODUCTIONS + 1):
        crop = random.choice(crop_types)
        variety = random.choice(varieties[crop])
        planting_date = fake.date_between(start_date='-2y', end_date='-6m')
        harvest_date = planting_date + timedelta(days=random.randint(90, 160))
        yield_kg = random.uniform(1000, 10000)
        avg_yield = yield_kg / random.uniform(0.5, 2.5)  # yield per hectare
        expected_price = round(random.uniform(0.3, 1.5), 2)
        records.append({
            'production_id': i,
            'farm_name': fake.company(),
            'field_id': f'F-{random.randint(100, 999)}',
            'field_location': fake.city(),
            'crop_type': crop,
            'crop_variety': variety,
            'planting_date': planting_date,
            'harvest_date': harvest_date,
            'fertilizer_type': random.choice(fertilizers),
            'fertilizer_quantity_kg': round(random.uniform(50, 300), 1),
            'pesticide_type': random.choice(pesticides),
            'pesticide_quantity_ltr': round(random.uniform(5, 30), 1),
            'irrigation_type': random.choice(irrigation_types),
            'irrigation_quantity_ltr': round(random.uniform(500, 2000), 1),
            'total_yield_kg': round(yield_kg, 1),
            'avg_yield_per_hectare': round(avg_yield, 1),
            'soil_ph': round(random.uniform(5.5, 7.5), 2),
            'rainfall_mm': round(random.uniform(200, 800), 1),
            'avg_temperature_c': round(random.uniform(12, 26), 1),
            'labor_hours': round(random.uniform(50, 200), 1),
            'number_of_workers': random.randint(2, 10),
            'machinery_used': random.choice(['Tractor', 'Harvester', 'Plough']),
            'expected_market_price_per_kg': expected_price,
            'expected_total_revenue_eur': round(expected_price * yield_kg, 2)
        })
    return pd.DataFrame(records)

def generate_sales_data(production_df):
    records = []
    for i in range(1, NUM_SALES + 1):
        prod = production_df.sample(1).iloc[0]
        crop = prod['crop_type']
        variety = prod['crop_variety']
        market_price = prod['expected_market_price_per_kg']
        actual_price = round(market_price * random.uniform(0.9, 1.1), 2)
        quantity = round(random.uniform(200, 2000), 1)
        revenue = round(quantity * actual_price, 2)
        discount = round(random.uniform(0, 50), 2)
        shipping = round(random.uniform(10, 100), 2)
        net = revenue - discount - shipping
        records.append({
            'sales_id': i,
            'production_id': prod['production_id'],
            'crop_type': crop,
            'crop_variety': variety,
            'buyer_name': fake.company(),
            'buyer_type': random.choice(buyer_types),
            'buyer_region': random.choice(regions),
            'channel_type': random.choice(channel_types),
            'transaction_date': fake.date_between(start_date=prod['harvest_date'], end_date='today'),
            'quantity_sold_kg': quantity,
            'unit_price_eur': actual_price,
            'total_revenue_eur': revenue,
            'discount_applied_eur': discount,
            'net_revenue_eur': round(net, 2),
            'shipping_cost_eur': shipping,
            'profit_margin_pct': round(random.uniform(5, 25), 2),
            'market_price_per_kg': market_price,
            'price_variance': round(actual_price - market_price, 2)
        })
    return pd.DataFrame(records)

# now generating the data
production_df = generate_production_data()
sales_df = generate_sales_data(production_df)

production_df.to_csv('fact_production.csv', index=False)
sales_df.to_csv('fact_sales.csv', index=False)

print("Production Sample:")
print(production_df.head())
print("\nSales Sample:")
print(sales_df.head())

