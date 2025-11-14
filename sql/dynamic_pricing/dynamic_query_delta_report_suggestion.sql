-- =============================================================
-- Overview: Implements the delta-report query requested during review. Layered
-- CTEs carve out the target category, enrich it with pricing context, and then
-- surface the final rowset ordered by purchase date for easy export.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about testing the data on a query(as suggested to be added in delta report)


-- Add indexes to optimize performance
CREATE INDEX idx_orders_order_id ON Orders (order_id);
CREATE INDEX idx_product_category ON Product (category_id);
CREATE INDEX idx_price_product ON Price (product_id, seller_id);
CREATE INDEX idx_orders_date ON Orders (order_date);
CREATE INDEX idx_seller ON Seller (seller_id);

WITH FilteredCategory AS (
    SELECT DISTINCT Product_Category.product_id
    FROM Product_Category
    WHERE Product_Category.category_id = 1
),
ProductDetails AS (
    SELECT 
        Product.product_id,
        Product.title AS product_name,
        Brand.brand_name,
        Price.price,
        Price.price_source,
        AVG(Price.price) OVER (PARTITION BY Price.product_id) AS avg_product_price
    FROM Product
    LEFT JOIN Brand ON Product.brand_id = Brand.brand_id
    JOIN Price ON Product.product_id = Price.product_id
    WHERE Product.product_id IN (SELECT FilteredCategory.product_id FROM FilteredCategory)
),
FinalDetails AS (
    SELECT 
        Orders.order_id,
        Orders.order_date,
        ProductDetails.product_name,
        Price.seller_id,
        ProductDetails.brand_name,
        ProductDetails.price,
        ProductDetails.price_source,
        ProductDetails.avg_product_price
    FROM Orders
    JOIN Order_Detail ON Orders.order_id = Order_Detail.order_id
    JOIN ProductDetails ON Order_Detail.product_id = ProductDetails.product_id
    JOIN Price ON ProductDetails.product_id = Price.product_id
    LEFT JOIN Seller ON Price.seller_id = Seller.seller_id
)
SELECT 
    FinalDetails.order_id,
    FinalDetails.product_name,
    FinalDetails.seller_id,
    FinalDetails.brand_name,
    FinalDetails.price,
    FinalDetails.price_source,
    FinalDetails.avg_product_price
FROM FinalDetails
ORDER BY FinalDetails.order_date, FinalDetails.product_name;