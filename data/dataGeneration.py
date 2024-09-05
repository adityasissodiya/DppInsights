import pandas as pd
import numpy as np
import os

def generate_pseudo_product_data(num_products=1000, save_path='data/processed/pseudo_product_data.csv'):
    # Product Type categories
    product_types = ['Electronics', 'Appliances', 'Clothing']
    # Material composition categories
    material_types = ['Plastic', 'Metal', 'Glass']
    # End-of-life options
    end_of_life_options = ['Recyclable', 'Hazardous']
    
    # Generate pseudo data
    data = {
        'ProductID': np.arange(1, num_products + 1),
        'ProductName': [f"Product_{i}" for i in range(1, num_products + 1)],
        'Manufacturer': np.random.choice(['CompanyA', 'CompanyB', 'CompanyC'], num_products),
        'YearOfManufacture': np.random.randint(2000, 2024, num_products),
        'ProductType': np.random.choice(product_types, num_products),
        
        # Material composition as percentages
        'MaterialPlastic': np.random.randint(0, 100, num_products),
        'MaterialMetal': np.random.randint(0, 100, num_products),
        'MaterialGlass': np.random.randint(0, 100, num_products),
        
        # Ensure the sum of material percentages is <= 100
        'RecyclabilityScore': np.random.randint(1, 101, num_products),
        
        # Targets for regression
        'CO2Emissions': np.random.uniform(50, 1000, num_products),
        'RepairCost': np.random.uniform(50, 1000, num_products),
        
        # Lifecycle attributes
        'RepairFrequency': np.random.randint(0, 10, num_products),
        'RepairabilityScore': np.random.randint(1, 11, num_products),
        'EndOfLifeOption': np.random.choice(end_of_life_options, num_products),
        
        # Additional features
        'SupplierLocation': np.random.choice(['USA', 'China', 'Germany', 'Sweden'], num_products),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure material percentages don't exceed 100
    df['MaterialSum'] = df[['MaterialPlastic', 'MaterialMetal', 'MaterialGlass']].sum(axis=1)
    df['MaterialPlastic'] = df['MaterialPlastic'] / df['MaterialSum'] * 100
    df['MaterialMetal'] = df['MaterialMetal'] / df['MaterialSum'] * 100
    df['MaterialGlass'] = df['MaterialGlass'] / df['MaterialSum'] * 100
    df.drop(columns=['MaterialSum'], inplace=True)

    # Ensure directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV for future use
    df.to_csv(save_path, index=False)

    return df

# Generate and save dataset
df = generate_pseudo_product_data()
print(df.head())
