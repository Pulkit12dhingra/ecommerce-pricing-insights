-- =============================================================
-- Overview: Defines the star-schema (sales fact + time dimension) and the temp
-- staging helpers used to move curated transactional data into analytics-ready
-- shapes. The surrounding queries document why each index, staging table, and
-- cleanup step exists before populating the main `sales` table.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about creating tables for a data warehouse star schema

-- check data types for all tables (these are dimension tables)
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'product';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'users';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'seller';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'orders';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'order_detail';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'price';

-- creating a table called 'sales' a fact table
CREATE TABLE sales (
    sales_id SERIAL PRIMARY KEY,        -- Unique identifier for each sales record
    product_id VARCHAR NOT NULL,        -- Product ID (referencing Product table)
    user_id INTEGER NOT NULL,           -- User ID (referencing Users table)
    seller_id INTEGER NOT NULL,         -- Seller ID (referencing Seller table)
    order_id BIGINT NOT NULL,           -- Order ID (referencing Orders table)
    price NUMERIC(10, 2),               -- Price of the product
    quantity INTEGER,                   -- Quantity sold
    shipping_price NUMERIC(10, 2),      -- Shipping price for the order
    total_price NUMERIC(10, 2),         -- Total price (calculated as price * quantity + shipping_price)
    time_id SERIAL NOT NULL,            -- Time Dimension Reference
    FOREIGN KEY (product_id) REFERENCES product (product_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (seller_id) REFERENCES seller (seller_id),
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

-- creating a table called 'time' a dimension table
CREATE TABLE time (
    time_id SERIAL PRIMARY KEY,         
    date DATE NOT NULL,                 
    day INTEGER,                        
    month INTEGER,                      
    year INTEGER,                       
    quarter INTEGER                     
);

-- Insert into 'time' table
INSERT INTO time (date, day, month, year, quarter)
SELECT DISTINCT 
    order_date::date AS date,
    EXTRACT(DAY FROM order_date) AS day,
    EXTRACT(MONTH FROM order_date) AS month,
    EXTRACT(YEAR FROM order_date) AS year,
    CEIL(EXTRACT(MONTH FROM order_date) / 3.0) AS quarter
FROM orders;

SELECT * FROM time;

CREATE INDEX idx_order_detail_order_id ON order_detail(order_id);
CREATE INDEX idx_orders_order_id ON orders(order_id);
CREATE INDEX idx_price_product_id ON price(product_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);

SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('order_detail', 'orders', 'price', 'sales');

DROP INDEX IF EXISTS sales_idx;

ALTER TABLE sales DISABLE TRIGGER ALL;

INSERT INTO seller (seller_id, seller_name, seller_rating)
VALUES (0, 'Unknown Seller', 0.0);

SELECT *
FROM price
WHERE seller_id IS NULL;

UPDATE price
SET seller_id = 0
WHERE seller_id IS NULL;

DELETE FROM temp_sales_staging;
DROP TABLE temp_sales_staging;

CREATE TEMP TABLE temp_sales_staging AS
(
    (SELECT
        od.product_id::VARCHAR AS product_id,  
        o.user_id::INTEGER AS user_id,
        p.seller_id::INTEGER AS seller_id,
        o.order_id::INTEGER AS order_id,
        p.price::NUMERIC(10,2) AS price,
        od.quantity::INTEGER AS quantity,
        od.shipping_price::NUMERIC(10,2) AS shipping_price,
        (p.price * od.quantity + od.shipping_price)::NUMERIC(10,2) AS total_price,
        t.time_id::INTEGER AS time_id,
        p.price_source::TEXT AS price_source
    FROM order_detail od
    JOIN orders o ON od.order_id = o.order_id
    JOIN price p ON od.product_id = p.product_id
    JOIN time t ON o.order_date::date = t.date
    WHERE p.price_source = 'kaggle'
    LIMIT 50000)

 	UNION ALL

    (SELECT
        od.product_id::VARCHAR AS product_id,  
        o.user_id::INTEGER AS user_id,
        p.seller_id::INTEGER AS seller_id,
        o.order_id::INTEGER AS order_id,
        p.price::NUMERIC(10,2) AS price,
        od.quantity::INTEGER AS quantity,
        od.shipping_price::NUMERIC(10,2) AS shipping_price,
        (p.price * od.quantity + od.shipping_price)::NUMERIC(10,2) AS total_price,
        t.time_id::INTEGER AS time_id,
        p.price_source::TEXT AS price_source
    FROM order_detail od
    JOIN orders o ON od.order_id = o.order_id
    JOIN price p ON od.product_id = p.product_id
    JOIN time t ON o.order_date::date = t.date
    WHERE p.price_source = 'ebay'
    LIMIT 50000)
);

SELECT COUNT(*) FROM temp_sales_staging;
SELECT * FROM temp_sales_staging LIMIT 10;

DROP INDEX IF EXISTS idx_sales_product_id;
DROP INDEX IF EXISTS idx_sales_time_id;

CREATE INDEX idx_sales_product_id ON sales(product_id);
CREATE INDEX idx_sales_time_id ON sales(time_id);

SELECT COUNT(*) FROM sales;
SELECT * FROM sales LIMIT 10;


SELECT * FROM sales LIMIT 10;

-----------------------------------------
-- Delete all rows from the sales table
DELETE FROM sales;
DELETE FROM temp_sales_sample;
drop table temp_sales_sample;

-- Create a temporary table to hold a sample of data from temp_sales_staging
CREATE TABLE sales_sample AS
SELECT *
FROM temp_sales_staging
LIMIT 100000; 

SELECT * FROM sales_sample;

-- Insert data into sales table
INSERT INTO sales (product_id, user_id, seller_id, order_id, price, quantity, shipping_price, total_price, time_id)
SELECT 
    product_id, 
    user_id, 
    seller_id, 
    order_id, 
    price, 
    quantity, 
    shipping_price, 
    total_price, 
    time_id
FROM temp_sales_sample;

SELECT * FROM sales WHERE user_id = -1;
SELECT * FROM sales;
