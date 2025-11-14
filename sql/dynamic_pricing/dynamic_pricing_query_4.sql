-- =============================================================
-- Overview: Grab bag of advanced SQL patterns (joins, windowing, OLAP rollups)
-- used to vet the transactional layer. The sections that follow walk through
-- each requirement so the same script can double as lecture/demo material.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about testing the data on topics such as
--- joins and subqueries, aggregates and grouping , OLAP and grouping functions


-- Working analysis 
-- Joins and Subqueries

-- Section 1 
-- 1.Write a SELECT statement that lists the following details: order ID, product name, seller ID, brand name, from price table, price, and platform source (e.g., eBay or Kaggle).
-- Include the average product price across all platforms for the same product. 
-- Filter the results by a specific product category (using ID). 

SELECT 
    Orders.order_id,
    Product.title AS product_name,
    Seller.seller_id,
    Brand.brand_name,
    Price.price,
    Price.price_source AS platform_source,
    AVG(Price.price) OVER (PARTITION BY Price.product_id) AS avg_product_price
FROM Orders
JOIN Order_Detail ON Orders.order_id = Order_Detail.order_id
JOIN Product ON Order_Detail.product_id = Product.product_id
JOIN Product_Category ON Product.product_id = Product_Category.product_id
JOIN Category ON Product_Category.category_id = Category.category_id
JOIN Price ON Product.product_id = Price.product_id
LEFT JOIN Seller ON Price.seller_id = Seller.seller_id
LEFT JOIN Brand ON Product.brand_id = Brand.brand_id
WHERE Product_Category.category_id = 1 
ORDER BY Orders.order_date, Product.title;

-- Section 2
-- Query to list all orders for a particular date, product category, and brand
SELECT 
    Orders.order_date::date AS order_date, -- Extract only the date portion of order_date
    Price.price_source AS platform_source, -- Source of the platform (eBay or Kaggle)
    Product.title AS product_name, -- Product name
    Brand.brand_name AS brand_name, -- Brand name of the product
    Category.category_name AS category_description -- Category description of the product
FROM Orders
-- Join Order_Detail to link orders and their details
INNER JOIN Order_Detail 
    ON Orders.order_id = Order_Detail.order_id
-- Join Product to fetch product details
INNER JOIN Product 
    ON Order_Detail.product_id = Product.product_id
-- Join Brand to fetch brand details
INNER JOIN Brand 
    ON Product.brand_id = Brand.brand_id
-- Join Product_Category to link products and their categories
INNER JOIN Product_Category 
    ON Product.product_id = Product_Category.product_id
-- Join Category to fetch category descriptions
INNER JOIN Category 
    ON Product_Category.category_id = Category.category_id
-- Join Price to get platform source details
INNER JOIN Price 
    ON Product.product_id = Price.product_id
-- Filter for a specific order date, product category, and brand
WHERE Orders.order_date = '2019-11-25' 
  AND Category.category_name = 'smartphone' 
  AND Brand.brand_name = 'samsung' 
-- Sort the results by order date and product name
ORDER BY Orders.order_date, Product.title;

-------------------------------------------------------

-- Aggregates and Grouping
-- section 1
-- Query to count the number of orders processed for each product and brand
SELECT 
    Product.title AS product_name, -- Product name
    Brand.brand_name AS brand_name, -- Brand name
    Price.price_source AS platform_source, -- Platform source (eBay or Kaggle)
    COUNT(DISTINCT Orders.order_id) AS total_orders -- Total number of orders for each product-brand combination
FROM Orders
-- Join Order_Detail to link orders with order details
INNER JOIN Order_Detail 
    ON Orders.order_id = Order_Detail.order_id
-- Join Product to fetch product details
INNER JOIN Product 
    ON Order_Detail.product_id = Product.product_id
-- Join Brand to fetch brand details
INNER JOIN Brand 
    ON Product.brand_id = Brand.brand_id
-- Join Price to get platform source
INNER JOIN Price 
    ON Product.product_id = Price.product_id
-- Group by product, brand, and platform source
GROUP BY Product.title, Brand.brand_name, Price.price_source
-- Filter for products and brands with 5 or more orders
HAVING COUNT(DISTINCT Orders.order_id) >= 5
-- Sort by total orders in descending order, then by product and brand name
ORDER BY total_orders DESC, Product.title, Brand.brand_name;

-- section 2
-- Query to return products and the number of orders they have by product category and month of order
SELECT 
    Product.title AS product_name, -- Product name
    Category.category_name AS product_category_description, -- Product category description
    TO_CHAR(Orders.order_date, 'YYYY-MM') AS month_of_order, -- Extract year and month of the order
    COUNT(DISTINCT Orders.order_id) AS total_orders -- Total number of orders for each product-category-month combination
FROM Orders
-- Join Order_Detail to link orders with products
INNER JOIN Order_Detail 
    ON Orders.order_id = Order_Detail.order_id
-- Join Product to fetch product details
INNER JOIN Product 
    ON Order_Detail.product_id = Product.product_id
-- Join Product_Category to link products to categories
INNER JOIN Product_Category 
    ON Product.product_id = Product_Category.product_id
-- Join Category to fetch category descriptions
INNER JOIN Category 
    ON Product_Category.category_id = Category.category_id
-- Filter for the specific year
WHERE EXTRACT(YEAR FROM Orders.order_date) = 2019 
-- Group by product name, product category, and month of order
GROUP BY Product.title, Category.category_name, TO_CHAR(Orders.order_date, 'YYYY-MM')
-- Sort by product category, month of order, and product name
ORDER BY Category.category_name, month_of_order, Product.title;

-- OLAP  & grouping 
-- section 1
-- Using ROLLUP Function
-- Query to calculate the number of products by category and brand with subtotals and grand totals using ROLLUP
SELECT 
    Category.category_name AS product_category, -- Product category description
    Brand.brand_name AS brand_name, -- Brand name
    COUNT(Product.product_id) AS total_products -- Total number of products
FROM Product
-- Join Product_Category to link products to categories
INNER JOIN Product_Category 
    ON Product.product_id = Product_Category.product_id
-- Join Category to fetch category descriptions
INNER JOIN Category 
    ON Product_Category.category_id = Category.category_id
-- Join Brand to fetch brand details
INNER JOIN Brand 
    ON Product.brand_id = Brand.brand_id
-- Group by category and brand using ROLLUP for subtotals and grand totals
GROUP BY ROLLUP (Category.category_name, Brand.brand_name)
-- Sort by category and brand name, with NULLs at the bottom for totals
ORDER BY 
    Category.category_name NULLS LAST, 
    Brand.brand_name NULLS LAST;

-- Using CUBE
-- Query to calculate the number of products by category and brand with subtotals and grand totals using CUBE
SELECT 
    Category.category_name AS product_category, -- Product category description
    Brand.brand_name AS brand_name -- Brand name
--    COUNT(Product.product_id) AS total_products -- Total number of products
FROM Product
-- Join Product_Category to link products to categories
INNER JOIN Product_Category 
    ON Product.product_id = Product_Category.product_id
-- Join Category to fetch category descriptions
INNER JOIN Category 
    ON Product_Category.category_id = Category.category_id
-- Join Brand to fetch brand details
INNER JOIN Brand 
    ON Product.brand_id = Brand.brand_id
-- Group by category and brand using CUBE for all possible subtotals and grand totals
--GROUP BY CUBE (Category.category_name, Brand.brand_name)
-- Sort by category and brand name, with NULLs at the bottom for totals
ORDER BY 
    Category.category_name NULLS LAST, 
    Brand.brand_name NULLS LAST;

-- section 2
-- Using RANK()
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
    RANK() OVER (ORDER BY total_orders DESC) AS rank -- Apply RANK for ranking
FROM CategoryOrders
ORDER BY rank, product_category, brand_name;

-- Using DENSE RANK()
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

--section 3
-- CTE to calculate average order value for each product category and brand
WITH CategoryAverage AS (
    SELECT 
        Category.category_name AS product_category, -- Product category
        Brand.brand_name AS brand_name, -- Associated brand name
        AVG(Price.price) AS average_order_value -- Calculate average order value
    FROM Orders
    INNER JOIN Order_Detail 
        ON Orders.order_id = Order_Detail.order_id -- Link orders to order details
    INNER JOIN Product 
        ON Order_Detail.product_id = Product.product_id -- Link order details to products
    INNER JOIN Product_Category 
        ON Product.product_id = Product_Category.product_id -- Link products to categories
    INNER JOIN Category 
        ON Product_Category.category_id = Category.category_id -- Link categories
    INNER JOIN Brand 
        ON Product.brand_id = Brand.brand_id -- Link products to brands
    INNER JOIN Price 
        ON Product.product_id = Price.product_id -- Link products to prices
    GROUP BY Category.category_name, Brand.brand_name -- Group by product category and brand
)
-- Main query to select top 3 categories with highest average order value
SELECT 
    product_category, 
    brand_name, 
    average_order_value
FROM CategoryAverage
ORDER BY average_order_value DESC -- Sort by average order value in descending order
LIMIT 3; -- Limit the results to the top 3

-- section 4
-- Query to find the 2 most expensive products for each brand
WITH ProductRanked AS (
    SELECT 
        Brand.brand_name AS brand_name, -- Brand name
        Product.title AS product_name, -- Product name
        Category.category_name AS category_name, -- Category name
        Price.price AS product_price, -- Product price
        ROW_NUMBER() OVER (PARTITION BY Brand.brand_name ORDER BY Price.price DESC) AS rank -- Rank products by price within each brand
    FROM Product
    INNER JOIN Product_Category 
        ON Product.product_id = Product_Category.product_id -- Link products to categories
    INNER JOIN Category 
        ON Product_Category.category_id = Category.category_id -- Link categories
    INNER JOIN Brand 
        ON Product.brand_id = Brand.brand_id -- Link products to brands
    INNER JOIN Price 
        ON Product.product_id = Price.product_id -- Link products to prices
)
-- Select only the top 2 products for each brand
SELECT 
    brand_name, 
    product_name, 
    category_name, 
    product_price
FROM ProductRanked	
WHERE rank <= 2 -- Limit results to the top 2 products per brand
ORDER BY brand_name, product_price DESC;

-- section 5
-- Query to calculate total products ordered for each brand and category
SELECT 
    Category.category_name AS product_category, -- Product category as row headings
    Brand.brand_name AS brand_name, -- Brand as column headings
    COUNT(Order_Detail.product_id) AS total_products -- Total products ordered
FROM Order_Detail
INNER JOIN Product 
    ON Order_Detail.product_id = Product.product_id -- Link order details with products
INNER JOIN Brand 
    ON Product.brand_id = Brand.brand_id -- Link products with brands
INNER JOIN Product_Category 
    ON Product.product_id = Product_Category.product_id -- Link products with categories
INNER JOIN Category 
    ON Product_Category.category_id = Category.category_id -- Link categories
GROUP BY Category.category_name, Brand.brand_name -- Group by product category and brand
ORDER BY Category.category_name, Brand.brand_name;

-- Pivot the results using a dynamic SQL approach for PostgreSQL
-- Step 1: Install tablefunc extension 
CREATE EXTENSION IF NOT EXISTS tablefunc;

-- Dynamically create column headers for each brand
SELECT *
FROM crosstab(
    $$
    SELECT 
        Category.category_name AS product_category, 
        Brand.brand_name AS brand_name, 
        COUNT(Order_Detail.product_id) AS total_products
    FROM Order_Detail
    INNER JOIN Product 
        ON Order_Detail.product_id = Product.product_id
    INNER JOIN Brand 
        ON Product.brand_id = Brand.brand_id
    INNER JOIN Product_Category 
        ON Product.product_id = Product_Category.product_id
    INNER JOIN Category 
        ON Product_Category.category_id = Category.category_id
    WHERE Brand.brand_name IN ('apple', 'samsung', 'asus') -- Filter for specific brands
    GROUP BY Category.category_name, Brand.brand_name
    ORDER BY Category.category_name, Brand.brand_name
    $$,
    $$
    VALUES ('apple'), ('samsung'), ('asus') -- Define the 3 brands explicitly
    $$
) AS pivot_table (
    product_category TEXT, 
    apple INT, 
    samsung INT, 
    asus INT
);
