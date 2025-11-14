-- =============================================================
-- Overview: Demonstrates denormalized history tracking via triggers. Sections
-- outline the audit table, per-operation triggers, and regression tests so you
-- can trace every order mutation and safely enable cascading deletes.
-- =============================================================
-- CS799 Advanced DBMS Term Project --- 
-- Pulkit Dhingra -- 
-- This code file is about testing the data on topic: Denormalization

CREATE TABLE order_history (
    history_id SERIAL PRIMARY KEY, -- Unique identifier for history records
    order_id INT NOT NULL, -- Reference to the original order
    user_id INT, -- User who placed the order
    product_name VARCHAR(255), -- Name of the product
    price NUMERIC(10, 2), -- Price of the product
    seller_name VARCHAR(255), -- Seller name
    platform_source VARCHAR(50), -- Source of the platform (e.g., eBay, Kaggle)
    shipping_price NUMERIC(10, 2), -- Shipping price of the order
    order_date DATE, -- Date of the order
    operation_type VARCHAR(50) NOT NULL, -- Type of operation (INSERT, UPDATE, DELETE)
    operation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Timestamp of operation
    changed_by VARCHAR(255) -- User/system who made the change
);

SELECT * FROM order_history;

-- Trigger for Insert:
CREATE OR REPLACE FUNCTION track_order_insert()
RETURNS TRIGGER AS $$
BEGIN
    -- Debugging logs
    RAISE NOTICE 'Trigger fired for order_id: %, user_id: %', NEW.order_id, NEW.user_id;

    INSERT INTO order_history (
        order_id, user_id, product_name, price, seller_name, platform_source,
        shipping_price, order_date, operation_type, changed_by
    )
    SELECT
        NEW.order_id, NEW.user_id, Product.title AS product_name,
        Price.price, Seller.seller_name, Price.price_source,
        Order_Detail.shipping_price, NEW.order_date,
        'INSERT', SESSION_USER
    FROM Order_Detail
    JOIN Product ON Order_Detail.product_id = Product.product_id
    JOIN Price ON Product.product_id = Price.product_id
    LEFT JOIN Seller ON Price.seller_id = Seller.seller_id
    WHERE Order_Detail.order_id = NEW.order_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;



DROP TRIGGER IF EXISTS order_insert_trigger ON Orders;

CREATE TRIGGER order_insert_trigger
AFTER INSERT ON Orders
FOR EACH ROW
EXECUTE FUNCTION track_order_insert();

-- Update
CREATE OR REPLACE FUNCTION track_order_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO order_history (
        order_id, user_id, product_name, price, seller_name, platform_source,
        shipping_price, order_date, operation_type, changed_by
    )
    SELECT
        NEW.order_id, NEW.user_id, Product.title AS product_name,
        Price.price, Seller.seller_name, Price.price_source,
        Order_Detail.shipping_price, NEW.order_date,
        'UPDATE', SESSION_USER
    FROM Order_Detail
    JOIN Product ON Order_Detail.product_id = Product.product_id
    JOIN Price ON Product.product_id = Price.product_id
    LEFT JOIN Seller ON Price.seller_id = Seller.seller_id
    WHERE Order_Detail.order_id = NEW.order_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS order_update_trigger ON Orders;

CREATE TRIGGER order_update_trigger
AFTER UPDATE ON orders
FOR EACH ROW EXECUTE FUNCTION track_order_update();

-- Delete
CREATE OR REPLACE FUNCTION track_order_delete()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO order_history (
        order_id, user_id, product_name, price, seller_name, platform_source,
        shipping_price, order_date, operation_type, changed_by
    )
    SELECT
        OLD.order_id, OLD.user_id, Product.title AS product_name,
        Price.price, Seller.seller_name, Price.price_source,
        Order_Detail.shipping_price, OLD.order_date,
        'DELETE', SESSION_USER
    FROM Order_Detail
    JOIN Product ON Order_Detail.product_id = Product.product_id
    JOIN Price ON Product.product_id = Price.product_id
    LEFT JOIN Seller ON Price.seller_id = Seller.seller_id
    WHERE Order_Detail.order_id = OLD.order_id;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS order_delete_trigger ON Orders;

CREATE TRIGGER order_delete_trigger
AFTER DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION track_order_delete();

-- Insert additional data to test the trigger
-- Insert a new order
INSERT INTO Orders (order_id, user_id, order_date)
VALUES (684081439, 575324361, '2024-01-03');

-- Check the `order_history` table
SELECT * FROM order_history WHERE operation_type = 'INSERT';

-- Update an existing order
UPDATE Orders
SET order_date = '2024-01-05'
WHERE order_id = 1;

-- Check the `order_history` table
SELECT * FROM order_history WHERE operation_type = 'UPDATE';

-- Delete an order
DELETE FROM Orders
WHERE order_id = 2;

-- Check the `order_history` table
SELECT * FROM order_history WHERE operation_type = 'DELETE';

SELECT * FROM order_history WHERE order_id = 3;

SELECT tgname, tgrelid::regclass, tgfoid::regprocedure
FROM pg_trigger
WHERE tgname = 'order_insert_trigger';

SELECT order_id, user_id, order_date FROM Orders;

SELECT * FROM Order_Detail ; 



INSERT INTO order_history (
    order_id, user_id, product_name, price, seller_name, platform_source,
    shipping_price, order_date, operation_type, changed_by
)
SELECT
    Orders.order_id, -- Include Orders table in FROM clause
    Orders.user_id,
    Product.title AS product_name,
    Price.price,
    Seller.seller_name,
    Price.price_source,
    Order_Detail.shipping_price,
    Orders.order_date,
    'INSERT' AS operation_type,
    SESSION_USER AS changed_by
FROM Orders
JOIN Order_Detail ON Orders.order_id = Order_Detail.order_id
JOIN Product ON Order_Detail.product_id = Product.product_id
JOIN Price ON Product.product_id = Price.product_id
LEFT JOIN Seller ON Price.seller_id = Seller.seller_id
WHERE Orders.order_id = 684081439; -- Replace with a valid order_id

INSERT INTO Orders (order_id, user_id, order_date)
VALUES (684081243, 575324361, '2024-01-03');

UPDATE Orders
SET order_date = '2024-11-28'
WHERE order_id = 684081439;

DELETE FROM Orders
WHERE order_id = 684081439;

-- Check order_history table
SELECT * FROM order_history;

-- Check order_history table
SELECT * FROM order_history WHERE operation_type = 'UPDATE';

-- Steps to Modify Foreign Key to Use ON DELETE CASCADE
-- Drop the existing foreign key constraint
ALTER TABLE Order_Detail
DROP CONSTRAINT order_detail_order_id_fkey;

-- Add a new foreign key constraint with ON DELETE CASCADE
ALTER TABLE Order_Detail
ADD CONSTRAINT order_detail_order_id_fkey
FOREIGN KEY (order_id)
REFERENCES Orders (order_id)
ON DELETE CASCADE;

DELETE FROM Orders
WHERE order_id = 684081439;

-- Verify the deletion
SELECT * FROM Order_Detail WHERE order_id = 684081439;
SELECT * FROM Orders WHERE order_id = 684081439;


CREATE OR REPLACE FUNCTION track_order_delete()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO order_history (
        order_id, user_id, product_name, price, seller_name, platform_source,
        shipping_price, order_date, operation_type, changed_by
    )
    SELECT
        OLD.order_id, OLD.user_id, Product.title AS product_name,
        Price.price, Seller.seller_name, Price.price_source,
        Order_Detail.shipping_price, OLD.order_date,
        'DELETE', SESSION_USER
    FROM Order_Detail
    JOIN Product ON Order_Detail.product_id = Product.product_id
    JOIN Price ON Product.product_id = Price.product_id
    LEFT JOIN Seller ON Price.seller_id = Seller.seller_id
    WHERE Order_Detail.order_id = OLD.order_id;

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS order_delete_trigger ON Orders;

CREATE TRIGGER order_delete_trigger
AFTER DELETE ON Orders
FOR EACH ROW
EXECUTE FUNCTION track_order_delete();

SELECT tgname, tgrelid::regclass, tgfoid::regprocedure
FROM pg_trigger
WHERE tgname = 'order_delete_trigger';

-- Check order_history table
SELECT * FROM order_history;
