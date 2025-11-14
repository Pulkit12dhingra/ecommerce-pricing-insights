-- =============================================================
-- Overview: Bootstraps the normalized OLTP schema that powers downstream
-- ingestion. Tables are created in dependency order so that each foreign key
-- is satisfied before it is referenced in later INSERT scripts. Run this once
-- on a clean database before loading any data.
-- =============================================================
-- Price Trends in e-commerce --- 
-- Pulkit Dhingra -- 
-- This code file is about creating tables based on the ERD diagram I designed

SELECT current_database();

-- create tables as per the erd diagram -- 
-- Create Brand Table
CREATE TABLE Brand (
    brand_id SERIAL PRIMARY KEY,
    brand_name VARCHAR(255) NOT NULL
);

----------------------------------------

-- Create Category Table
CREATE TABLE Category (
    category_id SERIAL PRIMARY KEY,
    category_type VARCHAR(255),
    category_name VARCHAR(255) NOT NULL,
    description TEXT
);

----------------------------------------

-- Create Product_Category Table
CREATE TABLE Product_Category (
    product_category_id SERIAL PRIMARY KEY,
    product_id INT NOT NULL,
    category_id INT NOT NULL,
    FOREIGN KEY (category_id) REFERENCES Category(category_id)
);

----------------------------------------

-- Create Product Table
CREATE TABLE Product (
    product_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    brand_id INT NOT NULL,
    product_category_id INT NOT NULL,
    FOREIGN KEY (brand_id) REFERENCES Brand(brand_id),
    FOREIGN KEY (product_category_id) REFERENCES Product_Category(product_category_id)
);

----------------------------------------

-- Create Seller Table
CREATE TABLE Seller (
    seller_id SERIAL PRIMARY KEY,
    seller_name VARCHAR(255) NOT NULL,
    seller_rating FLOAT
);

----------------------------------------

-- Create Price Table
CREATE TABLE Price (
    price_id SERIAL PRIMARY KEY,
    product_id INT NOT NULL,
    price FLOAT NOT NULL,
    price_source VARCHAR(255),
    seller_id INT,
    FOREIGN KEY (product_id) REFERENCES Product(product_id),
    FOREIGN KEY (seller_id) REFERENCES Seller(seller_id)
);

----------------------------------------

-- Create User Table
CREATE TABLE Users (
    user_id SERIAL PRIMARY KEY,
    user_name VARCHAR(255) NOT NULL,
    email VARCHAR(255)
);

----------------------------------------

-- Create User_Session Table
CREATE TABLE User_Session (
    session_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    event_id INT,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

----------------------------------------

-- Create Event Table
CREATE TABLE Events (
    event_id SERIAL PRIMARY KEY,
    event_type VARCHAR(255),
    event_time TIMESTAMP,
    session_id INT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES User_Session(session_id)
);

----------------------------------------

-- Create Order Table
CREATE TABLE Orders (
    order_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    order_date TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

----------------------------------------

-- Create Order_Detail Table
CREATE TABLE Order_Detail (
    order_detail_id SERIAL PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    price_id INT NOT NULL,
    quantity INT NOT NULL,
    shipping_price FLOAT,
    FOREIGN KEY (order_id) REFERENCES Orders(order_id),
    FOREIGN KEY (product_id) REFERENCES Product(product_id),
    FOREIGN KEY (price_id) REFERENCES Price(price_id)
);
