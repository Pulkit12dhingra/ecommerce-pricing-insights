-- =============================================================
-- Overview: Resets the transactional schema, samples 100k Kaggle rows, and
-- incrementally hydrates each dimension/fact table. Each logical block below
-- focuses on a single entity so the load order mirrors the FK dependencies
-- (brands/categories -> products -> prices -> users/orders -> events).
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about creating intermediate staging table(100k) as the kaggle data is 18M
-- Then inserting data into specific tables based on the created intermediate staging table


-- Disable foreign key constraints temporarily
SET session_replication_role = 'replica';

-- Truncate all tables to remove the inserted data
TRUNCATE TABLE Order_Detail RESTART IDENTITY CASCADE;
TRUNCATE TABLE Orders RESTART IDENTITY CASCADE;
TRUNCATE TABLE Product_Category RESTART IDENTITY CASCADE;
TRUNCATE TABLE Product RESTART IDENTITY CASCADE;
TRUNCATE TABLE Price RESTART IDENTITY CASCADE;
TRUNCATE TABLE Seller RESTART IDENTITY CASCADE;
TRUNCATE TABLE Category RESTART IDENTITY CASCADE;
TRUNCATE TABLE Brand RESTART IDENTITY CASCADE;
TRUNCATE TABLE Users RESTART IDENTITY CASCADE;
TRUNCATE TABLE user_session RESTART IDENTITY CASCADE;
TRUNCATE TABLE events RESTART IDENTITY CASCADE;

-- Drop the intermediate staging data table
DROP TABLE IF EXISTS intermediate_staging_data;

-- Re-enable foreign key constraints
SET session_replication_role = 'origin';

SELECT * FROM staging_ebay_data;
SELECT * FROM staging_kaggle_data LIMIT 10;

CREATE TABLE intermediate_staging_data AS
SELECT *
FROM staging_kaggle_data
LIMIT 100000;

-- Insert data from staging tables to their respective tables (previously created)
-- insert into brand table
INSERT INTO Brand (brand_name)
SELECT DISTINCT brand
FROM (
    SELECT brand FROM intermediate_staging_data
    UNION
    SELECT brand FROM staging_ebay_data
) AS all_brands
WHERE brand IS NOT NULL
ON CONFLICT DO NOTHING;

SELECT * FROM Brand;

--------------------------------------

-- insert into category table
INSERT INTO Category (category_name)
SELECT DISTINCT category_code
FROM (
    SELECT category_code FROM intermediate_staging_data
    UNION
    SELECT category_code FROM staging_ebay_data
) AS all_categories
WHERE category_code IS NOT NULL
ON CONFLICT DO NOTHING;

SELECT * FROM Category;

--------------------------------------

-- insert into seller table
INSERT INTO Seller (seller_name)
SELECT DISTINCT seller
FROM staging_ebay_data
WHERE seller IS NOT NULL
ON CONFLICT DO NOTHING;

SELECT * FROM Seller;

--------------------------------------

-- insert into product category table
INSERT INTO Product_Category (product_id, category_id)
SELECT DISTINCT 
    intermediate_staging_data.product_id,
    Category.category_id
FROM intermediate_staging_data
JOIN Category ON intermediate_staging_data.category_code = Category.category_name
WHERE intermediate_staging_data.product_id IS NOT NULL
  AND intermediate_staging_data.category_code IS NOT NULL;

-- Insert into Product_Category table from staging_ebay_data
INSERT INTO Product_Category (product_id, category_id)
SELECT DISTINCT 
    staging_ebay_data.product_id,
    Category.category_id
FROM staging_ebay_data
JOIN Category ON staging_ebay_data.category_code = Category.category_name
WHERE staging_ebay_data.product_id IS NOT NULL
  AND staging_ebay_data.category_code IS NOT NULL
ON CONFLICT DO NOTHING;

SELECT * FROM Product_Category;

--------------------------------------

-- Resolved: Issue with the Product table (foreign key constraints and data type format)
-- Identify All Tables Using product_id
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE column_name = 'product_id';

ALTER TABLE Order_Detail DROP CONSTRAINT order_detail_product_id_fkey;

ALTER TABLE Product DROP CONSTRAINT product_id; -- Drop the primary key if it exists

-- For Product table
ALTER TABLE Product ALTER COLUMN product_id TYPE character varying;

-- For Order_Detail table
ALTER TABLE Order_Detail ALTER COLUMN product_id TYPE character varying;

-- For Price table
ALTER TABLE Price ALTER COLUMN product_id TYPE character varying;

-- For Product_category table
ALTER TABLE product_category ALTER COLUMN product_id TYPE character varying;

-- Recreate the altered constraints
ALTER TABLE Order_Detail ADD CONSTRAINT order_detail_product_id_fkey
FOREIGN KEY (product_id)
REFERENCES Product (product_id);

-- insert into product table
INSERT INTO Product (product_id, title, brand_id, product_category_id)
SELECT DISTINCT 
    intermediate_staging_data.product_id,
    intermediate_staging_data.category_code AS title, -- Placeholder for title
    Brand.brand_id,
    Product_Category.product_category_id
FROM intermediate_staging_data
LEFT JOIN Brand ON intermediate_staging_data.brand = Brand.brand_name
LEFT JOIN Product_Category ON intermediate_staging_data.product_id = Product_Category.product_id
WHERE intermediate_staging_data.product_id IS NOT NULL
  AND Product_Category.product_category_id IS NOT NULL -- Exclude rows with NULL product_category_id
ON CONFLICT (product_id) DO NOTHING;

INSERT INTO Product (product_id, title, brand_id, product_category_id)
SELECT DISTINCT 
    staging_ebay_data.product_id,
    staging_ebay_data.title,
    Brand.brand_id,
    Product_Category.product_category_id
FROM staging_ebay_data
LEFT JOIN Brand ON staging_ebay_data.brand = Brand.brand_name
LEFT JOIN Product_Category ON staging_ebay_data.product_id = Product_Category.product_id
WHERE staging_ebay_data.product_id IS NOT NULL
  AND Product_Category.product_category_id IS NOT NULL -- Exclude rows with NULL product_category_id
ON CONFLICT (product_id) DO NOTHING;

SELECT * FROM Product;

--------------------------------------

-- insert price table
INSERT INTO Price (product_id, price, price_source, seller_id)
SELECT DISTINCT 
    intermediate_staging_data.product_id,
    intermediate_staging_data.price,
    'kaggle' AS price_source,
    CAST(NULL AS integer) -- Explicitly cast NULL as integer
FROM intermediate_staging_data
WHERE intermediate_staging_data.product_id IS NOT NULL
  AND intermediate_staging_data.price IS NOT NULL
ON CONFLICT DO NOTHING;

INSERT INTO Price (product_id, price, price_source, seller_id)
SELECT DISTINCT 
    staging_ebay_data.product_id,
    staging_ebay_data.price,
    'ebay' AS price_source,
    Seller.seller_id
FROM staging_ebay_data
LEFT JOIN Seller ON staging_ebay_data.seller = Seller.seller_name
WHERE staging_ebay_data.product_id IS NOT NULL
  AND staging_ebay_data.price IS NOT NULL
ON CONFLICT DO NOTHING;

SELECT * FROM Price;

--------------------------------------

-- insert data into users table

INSERT INTO Users (user_id)
SELECT DISTINCT 
    intermediate_staging_data.user_id
FROM intermediate_staging_data
WHERE intermediate_staging_data.user_id IS NOT NULL
ON CONFLICT DO NOTHING;

SELECT * FROM Users;

--------------------------------------

-- insert into orders table
-- Insert from intermediate_staging_data
INSERT INTO Orders (order_id, user_id, order_date)
SELECT DISTINCT
    nextval('order_id_seq') AS order_id, -- Generate order_id
    intermediate_staging_data.user_id, -- Use user_id from intermediate_staging_data
    intermediate_staging_data.event_time::date AS order_date -- Extract the date from event_time
FROM intermediate_staging_data
WHERE intermediate_staging_data.event_type = 'purchase' -- Filter only purchase events
ON CONFLICT DO NOTHING;

INSERT INTO Users (user_id)
VALUES (-1)
ON CONFLICT DO NOTHING;

-- Insert from staging_ebay_data
INSERT INTO Orders (order_id, user_id, order_date)
SELECT DISTINCT
    nextval('order_id_seq') AS order_id, -- Generate a unique order_id
    -1 AS user_id, -- Use the placeholder user_id
    NOW()::date AS order_date -- Use the current date as a default
FROM staging_ebay_data
WHERE staging_ebay_data.product_id IS NOT NULL -- Ensure product_id exists
  AND staging_ebay_data.seller IS NOT NULL; -- Ensure seller exists

UPDATE staging_ebay_data
SET event_type = 'purchase'
WHERE product_id IS NOT NULL;

SELECT * FROM Orders;
SELECT * 
FROM Orders
WHERE order_date = CURRENT_DATE; -- Check newly inserted rows

--------------------------------------

-- insert data into Order_detail table

INSERT INTO Order_Detail (order_detail_id, order_id, product_id, price_id, quantity, shipping_price)
SELECT DISTINCT
    nextval('order_detail_id_seq') AS order_detail_id, -- Generate unique order_detail_id
    Orders.order_id, -- Map order_id from Orders table
    intermediate_staging_data.product_id, -- Use product_id from intermediate_staging_data
    Price.price_id, -- Map price_id from Price table
	FLOOR(RANDOM() * 10 + 1) AS quantity,
    0.0 AS shipping_price -- Default shipping price (can be updated if available)
FROM intermediate_staging_data
JOIN Orders ON intermediate_staging_data.user_id = Orders.user_id -- Map order_id using user_id
JOIN Price ON intermediate_staging_data.product_id = Price.product_id -- Map price_id using product_id
WHERE intermediate_staging_data.product_id IS NOT NULL;

SELECT * FROM Order_Detail;

-- insert into order detail from staging_ebay_data
CREATE INDEX idx_staging_product_id ON staging_ebay_data(product_id);
CREATE INDEX idx_staging_seller ON staging_ebay_data(seller);
CREATE INDEX idx_seller_name ON Seller(seller_name);
CREATE INDEX idx_price_product_id ON Price(product_id);
CREATE INDEX idx_orders_user_id ON Orders(user_id);

WITH filtered_orders AS (
    SELECT order_id
    FROM Orders
    WHERE user_id = -1
)
INSERT INTO Order_Detail (order_detail_id, order_id, product_id, price_id, quantity, shipping_price)
SELECT DISTINCT
    nextval('order_detail_id_seq') AS order_detail_id,
    filtered_orders.order_id, -- Use pre-filtered orders
    staging_ebay_data.product_id,
    Price.price_id,
    FLOOR(RANDOM() * 10 + 1) AS quantity,
    staging_ebay_data.shipping_price
FROM staging_ebay_data
JOIN Seller ON staging_ebay_data.seller = Seller.seller_name
JOIN filtered_orders ON TRUE s-- Avoid re-filtering in the main query
JOIN Price ON staging_ebay_data.product_id = Price.product_id
WHERE staging_ebay_data.product_id IS NOT NULL
  AND staging_ebay_data.seller IS NOT NULL
  AND staging_ebay_data.shipping_price IS NOT NULL;

SELECT * FROM Order_detail LIMIT 10;	
	
--------------------------------------
-- insert into user_Session
INSERT INTO user_session (session_id, user_id, start_time, end_time, event_id)
SELECT DISTINCT
    ('x' || substr(md5(intermediate_staging_data.user_session::text), 1, 8))::bit(32)::int AS session_id, -- Hash UUID to integer
    intermediate_staging_data.user_id, -- Use user_id from intermediate_staging_data
    intermediate_staging_data.event_time AS start_time, -- Use event_time as session start time
    intermediate_staging_data.event_time + interval '1 hour' AS end_time, -- Assume sessions last 1 hour
    CAST(NULL AS integer) AS event_id -- Explicitly cast NULL as integer
FROM intermediate_staging_data
WHERE intermediate_staging_data.user_session IS NOT NULL -- Ensure session_id exists
  AND intermediate_staging_data.user_id IS NOT NULL -- Ensure user_id exists
ON CONFLICT (session_id) DO NOTHING; -- Avoid duplicate entries

SELECT * FROM user_session;	

----------------------------------------
CREATE SEQUENCE event_id_seq
START WITH 1
INCREMENT BY 1
NO MINVALUE
NO MAXVALUE
CACHE 1;

--------------------------------------

-- insert into events
INSERT INTO events (event_id, event_type, event_time, session_id)
SELECT DISTINCT
    nextval('event_id_seq') AS event_id, -- Generate unique event_id
    intermediate_staging_data.event_type, -- Use event_type from intermediate_staging_data
    intermediate_staging_data.event_time, -- Use event_time from intermediate_staging_data
    CAST(('x' || substr(md5(intermediate_staging_data.user_session::text), 1, 8))::bit(32)::int AS integer) AS session_id -- Hash UUID to integer
FROM intermediate_staging_data
WHERE intermediate_staging_data.event_type IS NOT NULL -- Ensure event_type exists
  AND intermediate_staging_data.event_time IS NOT NULL -- Ensure event_time exists
  AND intermediate_staging_data.user_session IS NOT NULL -- Ensure session_id exists
ON CONFLICT DO NOTHING; -- Avoid duplicate entries

SELECT * FROM events;	

UPDATE user_session
SET event_id = events.event_id
FROM events
WHERE user_session.session_id = events.session_id;

UPDATE Price
SET seller_id = seller.seller_id
FROM seller
WHERE Price.seller_id = seller.seller_id;