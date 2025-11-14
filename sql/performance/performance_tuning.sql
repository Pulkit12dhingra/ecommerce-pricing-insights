-- =============================================================
-- Overview: Documents the before/after tuning journey for a representative
-- order lookup query. Index maintenance, composite keys, and ranking examples
-- are grouped so you can measure impact (via EXPLAIN/plan cache) after each
-- optimization pass.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about performance tuning

-- Joins and Subqueries 
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

DROP INDEX idx_price_product_id;
DROP INDEX idx_order_date;
DROP INDEX idx_category_name;
DROP INDEX idx_brand_name;

-- Tuning Suggestions
-- Indexing

CREATE INDEX idx_orders_order_date ON Orders(order_date);
CREATE INDEX idx_order_detail_order_id ON Order_Detail(order_id);
CREATE INDEX idx_product_product_id ON Product(product_id);
CREATE INDEX idx_product_brand_id ON Product(brand_id);
CREATE INDEX idx_product_category_product_id ON Product_Category(product_id);
CREATE INDEX idx_price_product_id ON Price(product_id);
CREATE INDEX idx_category_category_name ON Category(category_name);
CREATE INDEX idx_brand_brand_name ON Brand(brand_name);

-- composite index
CREATE INDEX idx_order_order_id_date ON Orders(order_id, order_date);
CREATE INDEX idx_order_detail_order_product ON Order_Detail(order_id, product_id);
CREATE INDEX idx_product_category ON Product_Category(product_id, category_id);
CREATE INDEX idx_category_name ON Category(category_name);
CREATE INDEX idx_brand_name ON Brand(brand_name);

-- re-run the query
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
ORDER BY Orders.order_date, Product.title; -- composite

-- 
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

-- indexes
CREATE INDEX idx_order_detail_order_product ON Order_Detail (order_id, product_id);
CREATE INDEX idx_category_brand ON Product_Category (category_id, product_id);
CREATE INDEX idx_brand_product ON Brand (brand_id, brand_name);
CREATE INDEX idx_category_name ON Category (category_name);

