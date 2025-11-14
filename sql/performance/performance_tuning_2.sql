-- =============================================================
-- Overview: Stress-tests analytical patterns on sampled (100k) and larger (1M)
-- staging subsets. The notes in each block explain what the query measures
-- (e.g., category density, user funnel behavior) so you can compare runtimes
-- and evaluate indexing strategies for massive workloads.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about performance tuning - part 2 

SELECT * FROM staging_kaggle_data;

-- For 1M rows
CREATE TABLE staging_orders_1M AS
SELECT * FROM staging_kaggle_data
LIMIT 1000000;

SELECT * FROM staging_orders_1M;

-- For 100K rows
CREATE TABLE staging_orders_100K AS
SELECT * FROM staging_kaggle_data
LIMIT 100000;

SELECT * FROM staging_orders_100K;

-- Performance tuning checks
DROP INDEX idx_orders_order_date;
DROP INDEX idx_order_detail_order_id;
DROP INDEX idx_product_product_id;
DROP INDEX idx_product_brand_id;
DROP INDEX idx_product_category_product_id;
DROP INDEX idx_price_product_id;
DROP INDEX idx_category_category_name;
DROP INDEX idx_brand_brand_name;
DROP INDEX idx_price_product_id;
DROP INDEX idx_order_date;
DROP INDEX idx_category_name;
DROP INDEX idx_brand_name;

-- Query to list all orders for a particular date, product category, and brand
SELECT 
    staging_orders_100k.event_time, -- Event time
    staging_orders_100k.event_type, -- Type of the event (view, cart, etc.)
    staging_orders_100k.product_id, -- Product ID
    staging_orders_100k.brand, -- Brand name
    staging_orders_100k.category_code AS product_category, -- Product category
    staging_orders_100k.price AS product_price, -- Product price
    COUNT(subquery_category.user_id) AS total_users_in_category -- Total users in the category
FROM staging_orders_100k
-- Join a subquery to calculate category-level user counts
INNER JOIN (
    SELECT 
        staging_orders_100k.category_code,
        staging_orders_100k.user_id
    FROM staging_orders_100k
    WHERE staging_orders_100k.category_code = 'smartphone' -- Filter for smartphone category
) subquery_category
ON staging_orders_100k.category_code = subquery_category.category_code -- Join on category code
WHERE 
    staging_orders_100k.category_code = 'smartphone' -- Filter for smartphone category
    AND staging_orders_100k.brand = 'samsung' -- Filter for Samsung brand
    AND staging_orders_100k.event_type = 'view' -- Filter for 'view' events
    AND staging_orders_100k.event_time BETWEEN '2019-11-01' AND '2019-11-30' -- Filter by date range
GROUP BY 
    staging_orders_100k.event_time, staging_orders_100k.event_type, 
    staging_orders_100k.product_id, staging_orders_100k.brand, 
    staging_orders_100k.category_code, staging_orders_100k.price -- Group by relevant columns
ORDER BY 
    staging_orders_100k.event_time ASC; -- Sort by event time


--
EXPLAIN ANALYZE
SELECT 
    staging_orders_1M.event_time, -- Timestamp of the event
    staging_orders_1M.event_type, -- Type of the event (view, cart, etc.)
    staging_orders_1M.product_id, -- Product identifier
    staging_orders_1M.brand, -- Brand of the product
    staging_orders_1M.category_code AS product_category, -- Product category
    staging_orders_1M.price AS product_price, -- Price of the product
    staging_orders_1M.user_id, -- User identifier
    staging_orders_1M.user_session, -- Session identifier
    AVG(subquery_category.price) AS avg_category_price, -- Average price of products in the same category
    COUNT(DISTINCT subquery_category.user_id) AS unique_users_in_category -- Unique users interacting with the category
FROM staging_orders_1M
-- Join a subquery to calculate category-level aggregations
INNER JOIN (
    SELECT 
        staging_orders_1M.category_code,
        staging_orders_1M.user_id,
        staging_orders_1M.price
    FROM staging_orders_1M
    WHERE staging_orders_1M.category_code = 'smartphone' -- Filter for specific category
) subquery_category
ON staging_orders_1M.category_code = subquery_category.category_code -- Join based on category code
-- Join another subquery to find the most recent event time per user session
INNER JOIN (
    SELECT 
        staging_orders_1M.user_session,
        MAX(staging_orders_1M.event_time) AS latest_event_time
    FROM staging_orders_1M
    GROUP BY staging_orders_1M.user_session
) subquery_latest_session
ON staging_orders_1M.user_session = subquery_latest_session.user_session
   AND staging_orders_1M.event_time = subquery_latest_session.latest_event_time
WHERE 
    staging_orders_1M.category_code = 'smartphone' -- Filter for smartphone category
    AND staging_orders_1M.brand = 'samsung' -- Filter for Samsung brand
    AND staging_orders_1M.event_type = 'view' -- Filter for 'view' events
    AND staging_orders_1M.event_time BETWEEN '2019-11-01' AND '2019-11-30' -- Date range filter
GROUP BY 
    staging_orders_1M.event_time, staging_orders_1M.event_type, staging_orders_1M.product_id, 
    staging_orders_1M.brand, staging_orders_1M.category_code, staging_orders_1M.price, 
    staging_orders_1M.user_id, staging_orders_1M.user_session -- Group by relevant columns
ORDER BY 
    staging_orders_1M.event_time ASC; -- Sort by event time


--
EXPLAIN ANALYZE
SELECT 
    staging_orders_100k.event_time, -- Timestamp of the event
    staging_orders_100k.event_type, -- Type of the event (view, cart, etc.)
    staging_orders_100k.product_id, -- Product identifier
    staging_orders_100k.brand, -- Brand of the product
    staging_orders_100k.category_code AS product_category, -- Product category
    staging_orders_100k.price AS product_price, -- Price of the product
    staging_orders_100k.user_id, -- User identifier
    staging_orders_100k.user_session, -- Session identifier
    subquery_category.avg_price AS avg_category_price, -- Average price of products in the same category
    subquery_category.total_users AS unique_users_in_category -- Unique users interacting with the category
FROM staging_orders_100k
-- Join to calculate category-level statistics
INNER JOIN (
    SELECT 
        category_code,
        AVG(price) AS avg_price, -- Average price of the category
        COUNT(DISTINCT user_id) AS total_users -- Unique users in the category
    FROM staging_orders_100k
    GROUP BY category_code
) subquery_category
ON staging_orders_100k.category_code = subquery_category.category_code -- Join on category_code
WHERE 
    staging_orders_100k.category_code = 'smartphone' -- Filter for smartphone category
    AND staging_orders_100k.brand = 'samsung' -- Filter for Samsung brand
    AND staging_orders_100k.event_type = 'view' -- Filter for 'view' events
    AND staging_orders_100k.event_time BETWEEN '2019-11-01' AND '2019-11-30' -- Date range filter
ORDER BY 
    staging_orders_100k.event_time ASC; -- Sort by event time

--
SELECT 
    staging_orders_100k.product_id, -- Product ID
    staging_orders_100k.user_id, -- User ID
    staging_orders_100k.price, -- Price of the product
    subquery_product.avg_product_price AS avg_price_per_product, -- Average price for the product
    subquery_user.total_user_orders AS total_orders_per_user -- Total orders for the user
FROM staging_orders_100k
-- Join to calculate average price per product
INNER JOIN (
    SELECT 
        product_id,
        AVG(price) AS avg_product_price -- Calculate average price for each product
    FROM staging_orders_100k
    GROUP BY product_id
) subquery_product
ON staging_orders_100k.product_id = subquery_product.product_id -- Join on product_id
-- Join to calculate total orders per user
INNER JOIN (
    SELECT 
        user_id,
        COUNT(product_id) AS total_user_orders -- Count total orders for each user
    FROM staging_orders_100k
    GROUP BY user_id
) subquery_user
ON staging_orders_100k.user_id = subquery_user.user_id -- Join on user_id
WHERE 
    staging_orders_100k.price > 50 -- Example filter
ORDER BY 
    staging_orders_100k.price DESC; -- Sort by product price

SELECT 
    staging_orders_1M.product_id, -- Product ID
    staging_orders_1M.user_id, -- User ID
    staging_orders_1M.price, -- Price of the product
    subquery_product.avg_product_price AS avg_price_per_product, -- Average price for the product
    subquery_user.total_user_orders AS total_orders_per_user, -- Total orders for the user
    subquery_category.avg_category_price -- Average price for the product category
FROM staging_orders_1M
-- Join to calculate average price per product
INNER JOIN (
    SELECT 
        product_id,
        AVG(price) AS avg_product_price -- Calculate average price for each product
    FROM staging_orders_1M
    GROUP BY product_id
) subquery_product
ON staging_orders_1M.product_id = subquery_product.product_id -- Join on product_id
-- Join to calculate total orders per user
INNER JOIN (
    SELECT 
        user_id,
        COUNT(product_id) AS total_user_orders -- Count total orders for each user
    FROM staging_orders_1M
    GROUP BY user_id
) subquery_user
ON staging_orders_1M.user_id = subquery_user.user_id -- Join on user_id
-- Join to calculate average price per category
INNER JOIN (
    SELECT 
        category_id,
        AVG(price) AS avg_category_price -- Calculate average price for each category
    FROM staging_orders_1M
    GROUP BY category_id
) subquery_category
ON staging_orders_1M.category_id = subquery_category.category_id -- Join on category_id
WHERE 
    staging_orders_1M.price > 50 -- Example filter
ORDER BY 
    staging_orders_1M.price DESC; -- Sort by product price
