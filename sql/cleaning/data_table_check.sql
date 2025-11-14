-- =============================================================
-- Overview: Quick validation script that peeks into every transactional table
-- and patches obvious data quality gaps (e.g., missing descriptions, malformed
-- seller ratings). Use it after ingestion to spot-check integrity before the
-- warehouse build begins.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
--- check each table and fill null values

-- Select all records from each table
SELECT * FROM Brand;
SELECT * FROM Category;
SELECT * FROM Seller;
SELECT * FROM Product_Category;
SELECT * FROM Product;
SELECT * FROM Price;
SELECT * FROM Users;
SELECT * FROM Orders;
SELECT * FROM Order_Detail LIMIT 10;
SELECT * FROM User_Session;
SELECT * FROM Events;

-- insert description for the category 
-- Update descriptions for each category
UPDATE Category
SET description = 'Audio accessories such as headphones for personal use'
WHERE category_name = 'headphone';

UPDATE Category
SET description = 'Portable computing devices like tablets for entertainment and productivity'
WHERE category_name = 'tablet';

UPDATE Category
SET description = 'Mobile phones with advanced capabilities like smartphones for communication and applications'
WHERE category_name = 'smartphone';

-- verify
SELECT * FROM Category;

----------------------------------------------------------
-- verify seller rating 
SELECT * FROM Seller;

SELECT seller_name
FROM Seller
WHERE seller_name ~ '\(\d+\.\d+%\)'
LIMIT 10;

----------------------------------------------------------

