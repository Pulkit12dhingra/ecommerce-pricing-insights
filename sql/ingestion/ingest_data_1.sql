-- =============================================================
-- Overview: Loads raw Kaggle and eBay exports into lightweight staging tables
-- so they can be inspected, profiled, and transformed without touching the
-- production tables. Copy commands assume files already live in PostgreSQL's
-- data directory; update the absolute paths if the environment differs.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about ingesting data into staging tables

-- Creating a staging table and inserting 'Kaggle dataset'
-- columns: event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session

-- command to check current directory and I saved the files using bash in the current directory of PostgreSQL
SHOW data_directory;

-- ingest the kaggle dataset (18M records)
CREATE TABLE staging_kaggle_data (
    event_time TIMESTAMP,
    event_type VARCHAR(50),
    product_id VARCHAR(50),
    category_id BIGINT,
    category_code VARCHAR(50),
    brand VARCHAR(50),
    price NUMERIC,
    user_id BIGINT,
    user_session UUID
);

COPY staging_kaggle_data(event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session)
FROM '/Library/PostgreSQL/16/data/new_kaggle_dataset.csv'
DELIMITER ','
CSV HEADER;

SELECT * FROM staging_kaggle_data LIMIT 10;

-- ingest the ebay dataset (11k records)
CREATE TABLE staging_ebay_data (
    event_type VARCHAR(50),
    product_id VARCHAR(100) PRIMARY KEY,
    category_code VARCHAR(100),
    brand VARCHAR(100),
    price NUMERIC(10, 2),
    shipping_price NUMERIC(10, 2),
    seller VARCHAR(255),
    ratings NUMERIC(3, 2),
    title VARCHAR(255)
);

COPY staging_ebay_data(event_type, product_id, category_code, brand, price, shipping_price, seller, ratings, title)
FROM '/Library/PostgreSQL/16/data/new_ebay_dataset.csv'
DELIMITER ','
CSV HEADER;

SELECT * FROM staging_ebay_data LIMIT 10;