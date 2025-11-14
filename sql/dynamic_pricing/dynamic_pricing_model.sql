-- =============================================================
-- Overview: Builds the analytic table/materialized view that compares eBay vs
-- Kaggle pricing by category and brand. The calculated columns (average, median,
-- deltas, action flags) feed the dashboard and recommendation storyline.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about dynamic pricing model for the ecommerce project

-- Dynamic Pricing models

-- 1. Dynamic Pricing Model
CREATE TABLE dynamic_pricing_model (
    category_id INTEGER,
    brand_id INTEGER,
    avg_price_ebay NUMERIC(10, 2),
    avg_price_kaggle NUMERIC(10, 2),
    median_price_ebay NUMERIC(10, 2),
    median_price_kaggle NUMERIC(10, 2),
    price_difference NUMERIC(10, 2),
    suggested_action VARCHAR(50),
    top_brand_flag BOOLEAN,
    consistent_pricing_flag BOOLEAN
);

INSERT INTO dynamic_pricing_model (
    category_id,
    brand_id,
    avg_price_ebay,
    avg_price_kaggle,
    median_price_ebay,
    median_price_kaggle,
    price_difference,
    suggested_action,
    top_brand_flag,
    consistent_pricing_flag
)
SELECT 
    product_category.category_id,
    product.brand_id,

    -- Average prices for eBay and Kaggle
    AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) AS avg_price_ebay,
    AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END) AS avg_price_kaggle,

    -- Median prices for eBay and Kaggle
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price.price) 
        FILTER (WHERE price.price_source = 'ebay') AS median_price_ebay,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price.price) 
        FILTER (WHERE price.price_source = 'kaggle') AS median_price_kaggle,

    -- Price difference between eBay and Kaggle
    ABS(
        AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) -
        AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END)
    ) AS price_difference,

    -- Suggested action based on price difference
    CASE
        WHEN AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) <
             AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END)
            THEN 'Lower Price on eBay'
        ELSE 'Lower Price on kaggle'
    END AS suggested_action,

    -- Top brand/category flag
    CASE
        WHEN AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) < 100 OR
             AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END) < 100
        THEN TRUE
        ELSE FALSE
    END AS top_brand_flag,

    -- Consistent pricing flag (if price difference < 10)
    CASE
        WHEN ABS(
            AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) -
            AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END)
        ) < 10
        THEN TRUE
        ELSE FALSE
    END AS consistent_pricing_flag

FROM price
JOIN product ON price.product_id = product.product_id
JOIN product_category ON product.product_category_id = product_category.product_category_id
WHERE (product_category.category_id, product.brand_id) IN (
    SELECT 
        product_category.category_id,
        product.brand_id
    FROM price
    JOIN product ON price.product_id = product.product_id
    JOIN product_category ON product.product_category_id = product_category.product_category_id
    WHERE price.price_source = 'ebay'
    INTERSECT
    SELECT 
        product_category.category_id,
        product.brand_id
    FROM price
    JOIN product ON price.product_id = product.product_id
    JOIN product_category ON product.product_category_id = product_category.product_category_id
    WHERE price.price_source = 'kaggle'
)
GROUP BY product_category.category_id, product.brand_id;


SELECT * FROM dynamic_pricing_model_mv;

-- Create a Materialized View for Dynamic Pricing Model
-- Create the materialized view
CREATE MATERIALIZED VIEW dynamic_pricing_model_mv AS
SELECT 
    product_category.category_id,
    product.brand_id,

    -- Replace NULL with 0 for average prices
    COALESCE(AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END), 0) AS avg_price_ebay,
    COALESCE(AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END), 0) AS avg_price_kaggle,

    -- Replace NULL with 0 for median prices
    COALESCE(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price.price)
        FILTER (WHERE price.price_source = 'ebay'), 0
    ) AS median_price_ebay,
    COALESCE(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price.price)
        FILTER (WHERE price.price_source = 'kaggle'), 0
    ) AS median_price_kaggle,

    -- Calculate price difference and replace NULL with 0
    COALESCE(
        ABS(
            AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) -
            AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END)
        ), 0
    ) AS price_difference,

    -- Suggested action with a default value
    COALESCE(
        CASE
            WHEN AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) <
                 AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END)
            THEN 'Lower Price on eBay'
            ELSE 'Lower Price on Kaggle'
        END, 'No Action Needed'
    ) AS suggested_action,

    -- Top brand/category flag with default value FALSE
    COALESCE(
        CASE
            WHEN AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) < 100 OR
                 AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END) < 100
            THEN TRUE
            ELSE FALSE
        END, FALSE
    ) AS top_brand_flag,

    -- Consistent pricing flag with default value FALSE
    COALESCE(
        CASE
            WHEN ABS(
                AVG(CASE WHEN price.price_source = 'ebay' THEN price.price END) -
                AVG(CASE WHEN price.price_source = 'kaggle' THEN price.price END)
            ) < 10
            THEN TRUE
            ELSE FALSE
        END, FALSE
    ) AS consistent_pricing_flag

FROM price
JOIN product ON price.product_id = product.product_id
JOIN product_category ON product.product_category_id = product_category.product_category_id
WHERE (product_category.category_id, product.brand_id) IN (
    SELECT 
        product_category.category_id,
        product.brand_id
    FROM price
    JOIN product ON price.product_id = product.product_id
    JOIN product_category ON product.product_category_id = product_category.product_category_id
    WHERE price.price_source = 'ebay'
    INTERSECT
    SELECT 
        product_category.category_id,
        product.brand_id
    FROM price
    JOIN product ON price.product_id = product.product_id
    JOIN product_category ON product.product_category_id = product_category.product_category_id
    WHERE price.price_source = 'kaggle'
)
GROUP BY product_category.category_id, product.brand_id;

DROP MATERIALIZED VIEW IF EXISTS dynamic_pricing_model_mv CASCADE;



