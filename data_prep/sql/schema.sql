CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS amazon_products (
    asin TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    img_url TEXT NOT NULL,
    product_url TEXT NOT NULL,
    stars NUMERIC(3, 2) NOT NULL,
    reviews INTEGER NOT NULL,
    price NUMERIC(12, 2) NOT NULL,
    list_price NUMERIC(12, 2) NOT NULL,
    category_id INTEGER NOT NULL,
    is_best_seller BOOLEAN NOT NULL,
    bought_in_last_month INTEGER NOT NULL,
    main_category TEXT NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS amazon_product_title_embeddings (
    asin TEXT PRIMARY KEY REFERENCES amazon_products(asin) ON DELETE CASCADE,
    title_embedding vector(768) NOT NULL,
    model_name TEXT NOT NULL,
    embedding_created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_amazon_products_main_category
    ON amazon_products (main_category);

CREATE INDEX IF NOT EXISTS idx_amazon_products_category_id
    ON amazon_products (category_id);
