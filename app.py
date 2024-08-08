import requests
import json
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from pyairtable import Api
import os
from dotenv import load_dotenv
from io import StringIO

# Load environment variables
load_dotenv()

# Get Airtable API credentials and table names from environment variables
AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY')
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
AIRTABLE_UPLOADS_TABLE_NAME = os.getenv('AIRTABLE_UPLOADS_TABLE_NAME')
MAKE_WEBHOOK_URL = os.getenv('MAKE_WEBHOOK_URL')

# Initialize Airtable API
api = Api(AIRTABLE_API_KEY)
base = api.base(AIRTABLE_BASE_ID)
airtable_uploads = base.table(AIRTABLE_UPLOADS_TABLE_NAME)

app = Flask(__name__)

# Set pandas option to handle downcasting warning
pd.set_option('future.no_silent_downcasting', True)

# Function to fetch CSV from Airtable
def fetch_csv_from_airtable(record):
    try:
        attachments = record['fields'].get('CSV File', None)
        if not attachments:
            raise ValueError("CSV File field is missing in the record.")
        
        csv_url = attachments[0]['url']
        response = requests.get(csv_url)
        response.raise_for_status()
        
        # Try reading the CSV file with utf-8 encoding first
        try:
            csv_data = response.content.decode('utf-8')
            return pd.read_csv(StringIO(csv_data))
        except UnicodeDecodeError:
            # If utf-8 fails, try with ISO-8859-1 encoding
            csv_data = response.content.decode('ISO-8859-1')
            return pd.read_csv(StringIO(csv_data))
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

# Function to combine CSV files from Airtable
def combine_csv_files(selected_brand, batch_size=5):
    csv_records = airtable_uploads.all(formula=f"FIND('{selected_brand}', ARRAYJOIN({{Brand}}))")
    combined_data = pd.DataFrame()
    for i in range(0, len(csv_records), batch_size):
        batch_records = csv_records[i:i+batch_size]
        batch_data = [fetch_csv_from_airtable(record) for record in batch_records]
        batch_combined = pd.concat(batch_data, ignore_index=True)
        combined_data = pd.concat([combined_data, batch_combined], ignore_index=True)
    return combined_data

# Function to process combined data and generate reports
def process_combined_data(df):
    # Convert 'total_sales' to float with downcasting
    df.loc[:, 'total_sales'] = pd.to_numeric(df.loc[:, 'total_sales'], errors='coerce', downcast='float')

    # Drop rows where total_sales is 0 and product_title is NaN
    df = df.loc[~((df.loc[:, 'total_sales'] == 0) & (df.loc[:, 'product_title'].isna()))].copy()

    # Convert 'day' to datetime, dropping rows with invalid dates
    df['day'] = pd.to_datetime(df['day'], errors='coerce')
    df.dropna(subset=['day'], inplace=True)

    # Sort the dataframe by 'customer_email' and 'day' in ascending order, and 'product_title' in descending order
    df = df.sort_values(by=['customer_email', 'day', 'product_title'], ascending=[True, True, False])

    # Identify the first purchase order for each customer
    first_order = df.groupby('customer_email').first().reset_index()[['customer_email', 'order_id', 'day', 'product_title']]
    first_order.rename(columns={'order_id': 'first_order_id', 'day': 'first_purchase_day', 'product_title': 'first_product'}, inplace=True)

    # Merge first order details
    df = pd.merge(df, first_order, on='customer_email', how='left', suffixes=('', '_first'))

    # Ensure first_purchase_day is in datetime format
    df['first_purchase_day'] = pd.to_datetime(df['first_purchase_day'], errors='coerce')

    # Mark repeat purchases
    df.loc[:, 'is_first_purchase'] = df.loc[:, 'order_id'] == df.loc[:, 'first_order_id']

    # Forward fill the first purchase details
    df.loc[:, 'first_purchase_day'] = df['first_purchase_day'].ffill().bfill()
    df.loc[:, 'first_product'] = df['first_product'].ffill().bfill()

    # Create 'purchase_month' column
    df.loc[:, 'purchase_month'] = df['day'].dt.to_period('M')

    return df

# Function to generate cohort based on Month of first purchase
def generate_monthly_cohort(df):
    df['first_purchase_day'] = df.groupby('customer_email')['day'].transform('min')
    df['first_purchase_day'] = pd.to_datetime(df['first_purchase_day'], errors='coerce')
    df['cohort'] = df['first_purchase_day'].dt.to_period('M')
    return df

# Function to generate cohort based on First Product Purchased
def generate_first_product_cohort(df):
    df = df.dropna(subset=['customer_email'])
    df = df.sort_values(by=['customer_email', 'day'])
    df['first_product'] = df['first_product'].astype(str)
    df['cohort'] = df['first_product'].astype(str)
    return df

# Function to generate reports for a given cohort
def generate_reports_for_cohort(df, cohort_type):
    if cohort_type == 'Month':
        df['cohort'] = df['first_purchase_day'].dt.to_period('M')
    elif cohort_type == 'First Product Purchased':
        df['first_product'] = df['first_product'].astype(str)
        df['cohort'] = df['first_product']
    else:
        raise ValueError("Invalid cohort type")
    
    df['months_since_first_purchase'] = ((df['day'].dt.year - df['first_purchase_day'].dt.year) * 12 +
                                         (df['day'].dt.month - df['first_purchase_day'].dt.month))
    
    cohort_monthly_spend = df.groupby(['cohort', 'months_since_first_purchase'])['total_sales'].sum().reset_index()
    cohort_monthly_spend.columns = ['cohort', 'months_since_first_purchase', 'total_sales']
    cohort_monthly_spend['cumulative_total_spent'] = cohort_monthly_spend.groupby('cohort')['total_sales'].cumsum()
    
    cohort_sizes = df.groupby('cohort')['customer_email'].nunique().reset_index()
    cohort_sizes.columns = ['cohort', 'cohort_size']
    
    cohort_monthly_spend = pd.merge(cohort_monthly_spend, cohort_sizes, on='cohort')
    cohort_monthly_spend['avg_cumulative_total_spent'] = cohort_monthly_spend['cumulative_total_spent'] / cohort_monthly_spend['cohort_size']
    
    ltv = cohort_monthly_spend.pivot_table(index='cohort', columns='months_since_first_purchase', values='avg_cumulative_total_spent', fill_value=0)
    revenue = cohort_monthly_spend.pivot_table(index='cohort', columns='months_since_first_purchase', values='total_sales', fill_value=0)
    
    ltv = pd.concat([cohort_sizes.set_index('cohort'), ltv], axis=1)
    revenue = pd.concat([cohort_sizes.set_index('cohort'), revenue], axis=1)
    
    df.loc[:, 'is_repeat_purchase'] = ~df.loc[:, 'is_first_purchase']
    repeat_purchasers = df.loc[df.loc[:, 'is_repeat_purchase']].groupby(['cohort', 'months_since_first_purchase'])['customer_email'].nunique().reset_index()
    repeat_purchasers.columns = ['cohort', 'months_since_first_purchase', 'repeat_purchasers']
    repeat_purchasers = pd.merge(repeat_purchasers, cohort_sizes, on='cohort')
    repeat_purchasers['repeat_purchase_rate'] = repeat_purchasers['repeat_purchasers'] / repeat_purchasers['cohort_size']
    repeat_purchase_rate = repeat_purchasers.pivot_table(index='cohort', columns='months_since_first_purchase', values='repeat_purchase_rate', fill_value=0)
    
    repeat_purchase_rate = pd.concat([cohort_sizes.set_index('cohort'), repeat_purchase_rate], axis=1)
    
    return ltv, revenue, repeat_purchase_rate

# Function to generate and upload reports to the webhook
def generate_and_upload_reports(selected_brand):
    combined_data = combine_csv_files(selected_brand)
    combined_data = process_combined_data(combined_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    cohort_types = ['Month', 'First Product Purchased']

    for cohort_type in cohort_types:
        if cohort_type == 'Month':
            cohort_data = generate_monthly_cohort(combined_data)
        else:
            cohort_data = generate_first_product_cohort(combined_data)

        ltv, revenue, repeat_purchase_rate = generate_reports_for_cohort(cohort_data, cohort_type)
        
        reports = {
            'LTV': ltv,
            'Revenue': revenue,
            'Repeat Purchase Rate': repeat_purchase_rate
        }

        for report_type, report_data in reports.items():
            report_filename = f"{report_type}_{cohort_type}_{timestamp}.csv"
            report_path = os.path.join(tempfile.gettempdir(), report_filename)
            report_data.to_csv(report_path)

            with open(report_path, 'rb') as file:
                response = requests.post(
                    MAKE_WEBHOOK_URL,
                    files={'file': (report_filename, file, 'text/csv')},
                    data={
                        'report_type': report_type,
                        'cohort_type': cohort_type,
                        'start_date': combined_data['day'].min().strftime('%Y-%m-%d'),
                        'end_date': combined_data['day'].max().strftime('%Y-%m-%d'),
                    }
                )
                response.raise_for_status()

# Flask route to trigger report generation
@app.route('/generate_reports', methods=['POST'])
def generate_reports():
    try:
        data = request.get_json()
        selected_brand = data.get('brand')
        if not selected_brand:
            return jsonify({"error": "Brand not specified"}), 400
        
        generate_and_upload_reports(selected_brand)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)