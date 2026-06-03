# Shopping Product Recommendation Assistant

## Main Idea
Our team proposes to develop a shopping product recommendation assistant that converts natural language shopping requests into structured product queries, retrieves matching products from a product warehouse dataset, ranks them using relevance and business signals, and generates grounded recommendation explanations.

We would like to solve the business usecase where e-commerce platforms has a product warehouse with changing product inventory and strategic business goals. This assistant will help such a platform to query the product warehouse effectively based on users shopping requests, and generate taiored recommendations that align with the platform's business goals. 

For example, if a user asks for 

```
I want to buy a new 32 inch TV with good picture quality and a budget of $500. Can you recommend some options?
```

The assistant would convert this request into a structured product query that captures the key attributes (e.g., category: TV, size: 32 inch, features: good picture quality, price range: up to $500). It would then retrieve relevant products from the product warehouse, rank them based on relevance and business signals (e.g., inventory levels, profit margins), and generate an explanation for each recommendation that highlights how it meets the user's requirements.


## Key Project Outputs
1. Fine-tuned language model for converting natural language shopping requests into structured product queries.
2. Product database with relevant product metadata.
3. Vector embeddings for retrieving products based on context relevance.
4. Product ranking system that combines relevance and business signals.
5. Explanation generation module that provides grounded recommendations.

## Course Components

### Fine Tuning
We plan to fine-tune a lightweight pre-trained language model on a dataset of natural language shopping requests and their corresponding structured product queries using LoRA or similar techniques to efficiently generate accurate product queries.

### RAG and Vector Search
We plan to implement a Retrieval-Augmented Generation (RAG) system that retrieves relevant products from our product database using vector search based on the user's shopping request context. We plan to create vector embeddings for our product metadata to enable efficient and accurate product retrieval.

### Prompting 
We plan to design effective prompts to guide the language model in generating engaging and informative explanations for the recommended products. These prompts will be crafted to ensure that the generated explanations are grounded in the retrieved product information and align with the user's shopping intent.


## Project Resource Requirements

### Dataset
We would like to use the Amazon Product Dataset, which contains a large collection of product metadata, reviews, and ratings across various categories, as the basis for our product warehouse.
[Link to Dataset](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products)

### Database
Given the dataset resources above, we plan to build a scalable database solution such as PostgreSQL or MongoDB to store our product metadata and facilitate efficient querying and retrieval of product information.

### Vector Search Engine
To enable efficient retrieval of relevant products, we plan to use a vector search engine like FAISS or Pinecone based on vector embeddings of product metadata and user shopping requests.

### Pre-trained Language Model
We plan to use a lightweight pre-trained language model such as DistilBERT or GPT-2 for fine-tuning on our dataset of shopping requests and structured product queries.