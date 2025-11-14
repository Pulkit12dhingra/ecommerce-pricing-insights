# Price Trends in e-Commerce

[Blog Post](https://pulkit12dhingra.github.io/Blog/content/Pricing_Insights_Blog.html)

## Project Overview
This project analyzes price trends in e-commerce platforms using SQL, Python, and Tableau. It is organized for modularity, maintainability, and industrial standards.

## Directory Structure
```
/Price_Trends_in_e-Commerce
│
├── data/
│   ├── raw/
│   │   └── ebay_cleaned_dataset.csv
│   ├── processed/
│   │   └── ebay_cleaned_with_extracted_brands.csv
│
├── notebooks/
│   ├── apache_spark.ipynb
│   ├── Data_check.ipynb
│   └── Data_cleaning_code.ipynb
│
├── sql/
│   ├── ingestion/
│   │   ├── create_tables_2.sql
│   │   ├── ingest_data_1.sql
│   │   ├── ingest_data_each_table_100k.sql
│   │   ├── ingest_data_each_table_3.sql
│   │   └── insert_new_data.sql
│   ├── cleaning/
│   │   ├── data_table_check_v2.sql
│   │   └── data_warehouse_validation.sql
│   ├── schema/
│   │   └── data_warehouse_schema_v2.sql
│   ├── performance/
│   │   └── performance_tuning_2.sql
│   ├── procedures/
│   │   └── stored_procedures.sql
│   ├── dynamic_pricing/
│   │   ├── dynamic_pricing_model_v2.sql
│   │   ├── dynamic_pricing_query_4_v2.sql
│   │   ├── dynamic_pricing_query_5_v2.sql
│   │   └── dynamic_query_delta_report_suggestion_v2.sql
│
├── dashboard/
│   └── E-commerce-dashboard.twb
│
├── README.md
```

## Workflow
- **Data**: Raw and processed datasets for analysis.
- **Notebooks**: Data cleaning, validation, and scalable processing (Spark).
- **SQL**: Scripts for ingestion, cleaning, schema, performance tuning, procedures, and dynamic pricing analysis.
- **Dashboard**: Tableau workbook for visualization.

## Usage
1. Load and clean data using notebooks.
2. Ingest and validate data using SQL scripts.
3. Analyze price trends and dynamic pricing with advanced SQL queries.
4. Visualize results in Tableau dashboard.

## Notes
- All code and data are organized for clarity and maintainability.
- Only the latest versions of scripts are retained.
- Update paths in notebooks and scripts if you move files.

