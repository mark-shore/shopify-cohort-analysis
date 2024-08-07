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
        # Extract fields from the Airtable record
        print(f"Record fields: {record['fields']}")
        attachments = record['fields'].get('CSV File', None)
        if not attachments:
            raise ValueError("CSV File field is missing in the record.")
        
        # Get the URL of the CSV file
        csv_url = attachments[0]['url']
        print(f"CSV URL: {csv_url}")
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
def combine_csv_files():
    csv_records = airtable_uploads.all()
    combined_data = pd.DataFrame()
    for record in csv_records:
        csv_data = fetch_csv_from_airtable(record)
        combined_data = pd.concat([combined_data, csv_data])
    return combined_data

# Function to process combined data and generate reports
def process_combined_data(df):
    # Convert 'total_sales' to float
    df.loc[:, 'total_sales'] = pd.to_numeric(df.loc[:, 'total_sales'], errors='coerce')

    # Drop rows where total_sales is 0 and product_title is NaN
    df = df.loc[~((df.loc[:, 'total_sales'] == 0) & (df.loc[:, 'product_title'].isna()))]

    # Convert 'day' to datetime, dropping rows with invalid dates
    df.loc[:, 'day'] = pd.to_datetime(df.loc[:, 'day'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['day'])

    # Sort the dataframe by 'customer_email' and 'day' in ascending order, and 'product_title' in descending order
    df = df.sort_values(by=['customer_email', 'day', 'product_title'], ascending=[True, True, False])

    # Identify the first purchase order for each customer
    first_order = df.groupby('customer_email').first().reset_index()[['customer_email', 'order_id', 'day', 'product_title']]
    first_order = first_order.rename(columns={'order_id': 'first_order_id', 'day': 'first_purchase_day', 'product_title': 'first_product'})

    # Merge first order details
    df = pd.merge(df, first_order, on='customer_email', how='left', suffixes=('', '_first'))

    # Ensure first_purchase_day is in datetime format
    df.loc[:, 'first_purchase_day'] = pd.to_datetime(df.loc[:, 'first_purchase_day'], errors='coerce')

    # Mark repeat purchases
    df.loc[:, 'is_first_purchase'] = df.loc[:, 'order_id'] == df.loc[:, 'first_order_id']

    # Forward fill the first purchase details
    df.loc[:, 'first_purchase_day'] = df.loc[:, 'first_purchase_day'].ffill().bfill()
    df.loc[:, 'first_product'] = df.loc[:, 'first_product'].ffill().bfill()

    # Convert object columns to appropriate dtypes
    df = df.infer_objects(copy=False)

    # Create 'purchase_month' column
    df.loc[:, 'purchase_month'] = df.loc[:, 'day'].dt.to_period('M')

    # Ensure 'total_sales' is float
    df.loc[:, 'total_sales'] = df.loc[:, 'total_sales'].astype(float)

    return df

# Function to generate cohort based on Month of first purchase
def generate_monthly_cohort(df):
    df.loc[:, 'first_purchase_day'] = df.groupby('customer_email')['day'].transform('min')
    df.loc[:, 'first_purchase_day'] = pd.to_datetime(df.loc[:, 'first_purchase_day'], errors='coerce')
    df.loc[:, 'cohort'] = df.loc[:, 'first_purchase_day'].dt.to_period('M')
    return df

# Function to generate cohort based on First Product Purchased
def generate_first_product_cohort(df):
    df = df.dropna(subset=['customer_email'])
    df = df.sort_values(by=['customer_email', 'day'])
    # Explicitly cast to string dtype before assignment
    df.loc[:, 'first_product'] = df.loc[:, 'first_product'].astype(str)
    df.loc[:, 'cohort'] = df.loc[:, 'first_product'].astype(str)
    return df

# Function to generate reports for a given cohort
def generate_reports_for_cohort(df, cohort_type):
    # Determine cohort based on the type
    if cohort_type == 'Month':
        df.loc[:, 'cohort'] = df.loc[:, 'first_purchase_day'].dt.to_period('M')
    elif cohort_type == 'First Product Purchased':
        df.loc[:, 'first_product'] = df.loc[:, 'first_product'].astype(str)
        df.loc[:, 'cohort'] = df.loc[:, 'first_product']
    else:
        raise ValueError("Invalid cohort type")
    
    # Calculate time since the first purchase for each customer
    df['months_since_first_purchase'] = df.apply(lambda x: (x['day'].year - x['first_purchase_day'].year) * 12 + (x['day'].month - x['first_purchase_day'].month), axis=1)
    
    # Group by cohort and months since first purchase, and calculate metrics
    cohort_monthly_spend = df.groupby(['cohort', 'months_since_first_purchase'])['total_sales'].sum().reset_index()
    cohort_monthly_spend.columns = ['cohort', 'months_since_first_purchase', 'total_sales']
    cohort_monthly_spend['cumulative_total_spent'] = cohort_monthly_spend.groupby('cohort')['total_sales'].cumsum()
    
    # Calculate cohort sizes
    cohort_sizes = df.groupby('cohort')['customer_email'].nunique().reset_index()
    cohort_sizes.columns = ['cohort', 'cohort_size']
    
    # Merge cohort sizes into the spend data
    cohort_monthly_spend = pd.merge(cohort_monthly_spend, cohort_sizes, on='cohort')
    cohort_monthly_spend['avg_cumulative_total_spent'] = cohort_monthly_spend['cumulative_total_spent'] / cohort_monthly_spend['cohort_size']
    
    # Pivot tables for LTV and revenue
    ltv = cohort_monthly_spend.pivot_table(index='cohort', columns='months_since_first_purchase', values='avg_cumulative_total_spent', fill_value=0)
    revenue = cohort_monthly_spend.pivot_table(index='cohort', columns='months_since_first_purchase', values='total_sales', fill_value=0)
    
    # Add cohort sizes as the first column
    ltv = pd.concat([cohort_sizes.set_index('cohort'), ltv], axis=1)
    revenue = pd.concat([cohort_sizes.set_index('cohort'), revenue], axis=1)
    
    # Calculate repeat purchase rate
    df.loc[:, 'is_repeat_purchase'] = ~df.loc[:, 'is_first_purchase']
    repeat_purchasers = df.loc[df.loc[:, 'is_repeat_purchase']].groupby(['cohort', 'months_since_first_purchase'])['customer_email'].nunique().reset_index()
    repeat_purchasers.columns = ['cohort', 'months_since_first_purchase', 'repeat_purchasers']
    repeat_purchasers = pd.merge(repeat_purchasers, cohort_sizes, on='cohort')
    repeat_purchasers['repeat_purchase_rate'] = repeat_purchasers['repeat_purchasers'] / repeat_purchasers['cohort_size']
    repeat_purchase_rate = repeat_purchasers.pivot_table(index='cohort', columns='months_since_first_purchase', values='repeat_purchase_rate', fill_value=0)
    
    # Add cohort sizes as the first column
    repeat_purchase_rate = pd.concat([cohort_sizes.set_index('cohort'), repeat_purchase_rate], axis=1)
    
    return ltv, revenue, repeat_purchase_rate

# Function to generate and upload reports to the webhook
def generate_and_upload_reports(record_id):
    # Combine all CSV files
    combined_data = combine_csv_files()
    # Process the combined data
    combined_data = process_combined_data(combined_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Define cohort types
    cohort_types = ['Month', 'First Product Purchased']

    for cohort_type in cohort_types:
        # Generate cohort data based on the cohort type
        if cohort_type == 'Month':
            cohort_data = generate_monthly_cohort(combined_data)
        else:
            cohort_data = generate_first_product_cohort(combined_data)

        # Generate reports for the cohort
        ltv, revenue, repeat_purchase_rate = generate_reports_for_cohort(cohort_data, cohort_type)
        
        # Define reports to be generated
        reports = {
            'LTV': ltv,
            'Revenue': revenue,
            'Repeat Purchase Rate': repeat_purchase_rate
        }

        for report_type, report_data in reports.items():
            # Save each report to a CSV file
            report_filename = f"{report_type}_{cohort_type}_{timestamp}.csv"
            report_path = os.path.join(tempfile.gettempdir(), report_filename)
            report_data.to_csv(report_path)

            # Upload the report to the webhook
            with open(report_path, 'rb') as file:
                response = requests.post(
                    MAKE_WEBHOOK_URL,
                    files={'file': (report_filename, file, 'text/csv')},
                    data={
                        'report_type': report_type,
                        'cohort_type': cohort_type,
                        'start_date': combined_data['day'].min().strftime('%Y-%m-%d'),
                        'end_date': combined_data['day'].max().strftime('%Y-%m-%d'),
                        'record_id': record_id
                    }
                )
                response.raise_for_status()

# Flask route to trigger report generation
@app.route('/generate_reports', methods=['POST'])
def generate_reports():
    try:
        data = request.get_json()
        record_id = data['record_id']
        generate_and_upload_reports(record_id)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)