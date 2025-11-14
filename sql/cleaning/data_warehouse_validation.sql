-- =============================================================
-- Overview: Post-build verification for the warehouse layer. The script first
-- checks referential fidelity, then creates indexes, summary tables, and
-- materialized views that surface platform trends, category leaders, and
-- high-value orders. Run it after loading the sales fact to validate results.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about data warehouse validation

-- Data Validation after creation of the data warehouse
SELECT * FROM sales WHERE product_id IS NULL OR user_id IS NULL OR seller_id IS NULL;
SELECT COUNT(*) AS total_records, COUNT(DISTINCT product_id) AS unique_products FROM sales;

SELECT COUNT(*) AS mismatched_records
FROM sales s
LEFT JOIN product p ON s.product_id = p.product_id
WHERE p.product_id IS NULL;

-------------------------------------------------------

-- INDEX Optimization
CREATE INDEX idx_sales_product_id ON sales(product_id);
CREATE INDEX idx_sales_user_id ON sales(user_id);
CREATE INDEX idx_sales_time_id ON sales(time_id);

CREATE INDEX idx_sales_product_time ON sales(product_id, time_id);

EXPLAIN ANALYZE SELECT * FROM sales WHERE product_id = '100002292' AND time_id = 2;

-------------------------------------------------------

-- Data Aggregation
-- Summary tables
-- 1. Sales Summary by Platform and Time
CREATE TABLE sales_summary_platform_time AS
SELECT
    time.year,
    time.month,
    price.price_source,
    SUM(sales.total_price) AS total_revenue,
    COUNT(sales.order_id) AS total_orders,
    AVG(sales.price) AS average_price_per_order
FROM sales
JOIN time ON sales.time_id = time.time_id
JOIN price ON sales.product_id = price.product_id
GROUP BY time.year, time.month, price.price_source;

SELECT * FROM sales_summary_platform_time;

-- 2. Top selling products by category
CREATE TABLE top_selling_products_category AS
SELECT
    product_category.category_id,
    category.category_name,
    sales.product_id,
    product.title,
    COUNT(sales.order_id) AS total_orders,
    SUM(sales.quantity) AS total_quantity_sold,
    SUM(sales.total_price) AS total_revenue
FROM sales
JOIN product_category ON sales.product_id = product_category.product_id
JOIN category ON product_category.category_id = category.category_id
JOIN product ON sales.product_id = product.product_id
GROUP BY product_category.category_id, category.category_name, sales.product_id, product.title
ORDER BY category.category_name, total_quantity_sold DESC;

SELECT * FROM top_selling_products_category;

-- 3. Price Trends by Platform and Product
CREATE TABLE price_trends_platform_product AS
SELECT
    price.price_source,
    product.title,
    time.year,
    time.month,
    AVG(price.price) AS average_price,
    MAX(price.price) AS maximum_price,
    MIN(price.price) AS minimum_price
FROM price
JOIN product ON price.product_id = product.product_id
JOIN time ON time.time_id IN (
    SELECT DISTINCT sales.time_id
    FROM sales
    WHERE sales.product_id = price.product_id
)
GROUP BY price.price_source, product.title, time.year, time.month
ORDER BY price.price_source, product.title, time.year, time.month;

SELECT * FROM price_trends_platform_product;

-- 4. Category Revenue and Quantity Analysis
CREATE TABLE category_revenue_analysis AS
SELECT
    category.category_name,
    SUM(sales.total_price) AS total_revenue,
    SUM(sales.quantity) AS total_quantity_sold,
    AVG(sales.price) AS average_price_per_item
FROM sales
JOIN product_category ON sales.product_id = product_category.product_id
JOIN category ON product_category.category_id = category.category_id
GROUP BY category.category_name
ORDER BY total_revenue DESC;

SELECT * FROM category_revenue_analysis;

-- 5. Platform Performance Summary
CREATE TABLE platform_performance_summary AS
SELECT
    price.price_source,
    COUNT(sales.order_id) AS total_orders,
    SUM(sales.total_price) AS total_revenue,
    AVG(sales.shipping_price) AS average_shipping_price
FROM sales
JOIN price ON sales.product_id = price.product_id
GROUP BY price.price_source
ORDER BY total_revenue DESC;

SELECT * FROM platform_performance_summary;

-- 6. High value orders
CREATE TABLE high_value_orders AS
SELECT
    sales.order_id,
    orders.user_id,
    SUM(sales.total_price) AS order_value,
    COUNT(sales.product_id) AS total_products
FROM sales
JOIN orders ON sales.order_id = orders.order_id
JOIN users ON orders.user_id = users.user_id
GROUP BY sales.order_id, orders.user_id
HAVING SUM(sales.total_price) > 1000
ORDER BY order_value DESC;

SELECT * FROM high_value_orders;

-------------------------------------------------------

-- 1. Materialized View: Sales Summary by Platform and Time
CREATE MATERIALIZED VIEW sales_view_platform_time AS
SELECT
    time.year,
    time.month,
    price.price_source,
    SUM(sales.total_price) AS total_revenue,
    COUNT(sales.order_id) AS total_orders,
    AVG(sales.price) AS average_price_per_order
FROM sales
JOIN time ON sales.time_id = time.time_id
JOIN price ON sales.product_id = price.product_id
GROUP BY time.year, time.month, price.price_source;

-- 2. Materialized View: Product Performance by Category
CREATE MATERIALIZED VIEW product_performance_by_category AS
SELECT
    product_category.category_id,
    category.category_name,
    sales.product_id,
    product.title,
    COUNT(sales.order_id) AS total_orders,
    SUM(sales.quantity) AS total_quantity_sold,
    SUM(sales.total_price) AS total_revenue
FROM sales
JOIN product_category ON sales.product_id = product_category.product_id
JOIN category ON product_category.category_id = category.category_id
JOIN product ON sales.product_id = product.product_id
GROUP BY product_category.category_id, category.category_name, sales.product_id, product.title;

--3. Materialized View: Price Trends by Product and Platform
CREATE MATERIALIZED VIEW price_trends_product_platform AS
SELECT
    price.price_source,
    product.title,
    time.year,
    time.month,
    AVG(price.price) AS average_price,
    MAX(price.price) AS maximum_price,
    MIN(price.price) AS minimum_price
FROM price
JOIN product ON price.product_id = product.product_id
JOIN time ON time.time_id IN (
    SELECT DISTINCT sales.time_id
    FROM sales
    WHERE sales.product_id = price.product_id
)
GROUP BY price.price_source, product.title, time.year, time.month;

-- 4. Materialized View: Revenue by Category
CREATE MATERIALIZED VIEW revenue_by_category AS
SELECT
    category.category_name,
    SUM(sales.total_price) AS total_revenue,
    SUM(sales.quantity) AS total_quantity_sold,
    AVG(sales.price) AS average_price_per_item
FROM sales
JOIN product_category ON sales.product_id = product_category.product_id
JOIN category ON product_category.category_id = category.category_id
GROUP BY category.category_name;

SELECT price_source, year, month, SUM(total_revenue) AS revenue
FROM sales_view_platform_time
WHERE year = 2019
GROUP BY price_source, year, month
ORDER BY year, month;

-- 5 Materialized View: category_brand_order_ranking
CREATE MATERIALIZED VIEW category_brand_order_ranking AS
WITH CategoryOrders AS (
    SELECT 
        Category.category_name AS product_category, -- Product category description
        Brand.brand_name AS brand_name, -- Brand name
        COUNT(DISTINCT Orders.order_id) AS total_orders -- Total number of orders
    FROM Orders
    INNER JOIN Order_Detail 
        ON Orders.order_id = Order_Detail.order_id -- Link orders with order details
    INNER JOIN Product 
        ON Order_Detail.product_id = Product.product_id -- Link order details with products
    INNER JOIN Brand 
        ON Product.brand_id = Brand.brand_id -- Link products with brands
    INNER JOIN Product_Category 
        ON Product.product_id = Product_Category.product_id -- Link products with categories
    INNER JOIN Category 
        ON Product_Category.category_id = Category.category_id -- Link categories with product categories
    GROUP BY Category.category_name, Brand.brand_name -- Group by category and brand
)
SELECT 
    product_category,
    brand_name,
    total_orders,
    DENSE_RANK() OVER (ORDER BY total_orders DESC) AS dense_rank -- Apply DENSE_RANK for ranking
FROM CategoryOrders
ORDER BY dense_rank, product_category, brand_name;
