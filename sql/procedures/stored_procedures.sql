-- =============================================================
-- Overview: Central catalog of operational procedures—incremental loads, data
-- validation, daily summaries, and materialized-view refreshes. Each procedure
-- wraps the manual SQL steps used elsewhere so BI refreshes can be automated
-- or scheduled without retyping the underlying statements.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about stored procedures for data warehouse

-- Stored Procedures
-- 1. Incremental Data Loading
CREATE OR REPLACE PROCEDURE incremental_load_sales()
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO sales (product_id, user_id, seller_id, order_id, price, quantity, shipping_price, total_price, time_id)
    SELECT 
        temp.product_id,
        temp.user_id,
        temp.seller_id,
        temp.order_id,
        temp.price,
        temp.quantity,
        temp.shipping_price,
        temp.total_price,
        temp.time_id
    FROM sales_sample temp
    LEFT JOIN sales s
    ON temp.order_id = s.order_id
    WHERE s.order_id IS NULL; -- Only insert records that don't already exist
END;
$$;

-- Call this procedure when you have new data:
CALL incremental_load_sales();


-- 2. Data Validation
CREATE OR REPLACE PROCEDURE validate_and_insert_sales()
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO sales (product_id, user_id, seller_id, order_id, price, quantity, shipping_price, total_price, time_id)
    SELECT 
        product_id,
        user_id,
        seller_id,
        order_id,
        price,
        quantity,
        shipping_price,
        (price * quantity + shipping_price) AS total_price,
        time_id
    FROM sales_sample
    WHERE price > 0 AND quantity > 0; -- Ensure valid prices and quantities
END;
$$;

-- Call this procedure:
CALL validate_and_insert_sales();

-- 3.Pre-Calculated Summaries
CREATE TABLE daily_sales_summary (
    summary_date DATE PRIMARY KEY,       -- Date of the summary
    total_sales INT,                     -- Total number of sales
    total_revenue NUMERIC(12, 2)         -- Total revenue for the day
);


CREATE OR REPLACE PROCEDURE generate_daily_sales_summary()
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO daily_sales_summary (summary_date, total_sales, total_revenue)
    SELECT 
        time.date AS summary_date,
        COUNT(sales.order_id) AS total_sales,
        SUM(sales.total_price) AS total_revenue
    FROM sales
    JOIN time ON sales.time_id = time.time_id
    GROUP BY time.date
    ON CONFLICT (summary_date) 
    DO UPDATE SET 
        total_sales = EXCLUDED.total_sales,
        total_revenue = EXCLUDED.total_revenue;
END;
$$;

-- Call the procedure
CALL generate_daily_sales_summary();


-- Call this procedure:
CALL generate_daily_sales_summary();

-- 4. Refresh Materialized Views Stored Procedure
SELECT * FROM sales_view_platform_time;

CREATE UNIQUE INDEX idx_sales_summary_platform_time
ON sales_summary_platform_time (year, month, price_source);

CREATE UNIQUE INDEX idx_product_performance_by_category
ON product_performance_by_category (category_id, product_id);

CREATE UNIQUE INDEX idx_revenue_by_category
ON revenue_by_category (category_name);

CREATE UNIQUE INDEX idx_category_brand_order_ranking
ON category_brand_order_ranking (product_category, brand_name);

CREATE UNIQUE INDEX idx_dynamic_pricing_model_mv
ON dynamic_pricing_model_mv (category_id, brand_id);

CREATE OR REPLACE PROCEDURE refresh_materialized_views()
LANGUAGE plpgsql
AS $$
BEGIN
	REFRESH MATERIALIZED VIEW CONCURRENTLY dynamic_pricing_model_mv;
    REFRESH MATERIALIZED VIEW CONCURRENTLY product_performance_by_category;
	REFRESH MATERIALIZED VIEW CONCURRENTLY revenue_by_category;
	REFRESH MATERIALIZED VIEW CONCURRENTLY sales_summary_platform_time;
	REFRESH MATERIALIZED VIEW CONCURRENTLY category_brand_order_ranking;
    RAISE NOTICE 'Materialized views refreshed successfully!';
END;
$$;

-- Call this procedure:
CALL refresh_materialized_views();

REFRESH MATERIALIZED VIEW CONCURRENTLY sales_view_platform_time;