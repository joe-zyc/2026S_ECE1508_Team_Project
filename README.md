# Shopping Product Recommendation Assistant

This repository contains the ECE1508 team project: a shopping assistant that converts a natural-language request into structured search constraints, predicts the relevant Amazon product category, retrieves products using semantic vector search, and generates grounded recommendation explanations.

For example, a request such as _“I need a highly rated wireless gaming mouse under $50”_ is transformed into search fields such as product description, maximum price, minimum rating, category, and sort order. The system then combines those filters with embedding similarity to retrieve and explain suitable products.

## Repository structure

```text
.
├── data_prep/                  # PostgreSQL schema, ingestion, and embedding pipeline
├── fine_tune/                  # Category-classification model experiments
│   ├── jinhong/                # Qwen fine-tuning notebook
│   ├── jz/                     # Llama 3.2 1B fine-tuning notebook
│   └── ouyi/Gemma3_1b/         # Gemma training and query-pipeline notebooks
├── product_search/
│   ├── baselines/              # Query generation and zero-shot baselines
│   ├── final_query_constructor_pipeline_with_results.ipynb
│   └── requirement.txt         # Dependencies for the final notebook
├── proposal/                   # Project proposal and architecture sketch
├── report/                     # NeurIPS-style course report source
```

## Prerequisites

- Python 3.10 or newer; the final notebook was created with Python 3.12.
- JupyterLab, Jupyter Notebook, or Google Colab.
- Docker and Docker Compose for a local PostgreSQL/pgvector database.
- Access to the gated Hugging Face model `google/gemma-3-1b-it`.
- A trained Gemma LoRA category adapter.
- An OpenAI API key with access to the model configured in the final notebook.

## Quick start

### 1. Clone and create an environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r product_search/requirement.txt
```

### 2. Prepare the product database

The database layer expects PostgreSQL with the pgvector extension and two tables:

- `amazon_products`, containing Amazon product metadata;
- `amazon_product_title_embeddings`, containing one title embedding per product.

Detailed ingestion instructions are available in [`data_prep/README.md`](data_prep/README.md).

The default embedding model is `BAAI/bge-base-en-v1.5`. The model name and 768-dimensional output must remain consistent between ingestion and retrieval.

### 3. Prepare the category model

The final pipeline loads:

- base model: `google/gemma-3-1b-it`;
- a project-trained PEFT/LoRA adapter for the 16-class Amazon category task.

To train or inspect this model, open:

```text
fine_tune/ouyi/Gemma3_1b/train_gemma.ipynb
```

Researches with Llama and Qwen experiments are under `fine_tune/jz/` and `fine_tune/jinhong/`.

### 4. Final pipeline with Demo Results
The final pipeline notebook demonstrates the complete product search and recommendation process. It includes:
- Query construction and parsing
- Category prediction
- SQL generation and execution
- Product retrieval and display

Detailed results are available in `product_search/final_query_constructor_pipeline_with_results.ipynb`

## Usage

After running the notebook’s initialization cells, execute a complete personalized search:

```python
output = personalized_product_search(
    "I need a highly rated wireless gaming mouse under $50",
    top_x=5,
)

display(Markdown(output["message"]))
display(output["search"]["results"])
```

The returned dictionary contains:

- `search`: parsed constraints, predicted category, SQL, and retrieved products;
- `selected_products`: the displayed products and their generated reasons;
- `message`: a Markdown-formatted recommendation list.

For retrieval without generating recommendation text:

```python
search_output = product_search_pipeline(
    "A projector around $500 with at least 4.5 stars",
    limit=10,
)

display(search_output["results"])
```

## Fine-tuning and baselines

The `fine_tune/` directory records lightweight category-model experiments using Gemma, Llama, and Qwen. These notebooks include dataset preparation, LoRA training, evaluation, and model-specific outputs.

The `product_search/baselines/` notebooks provide baseline query generation and zero-shot classification results for comparison with the final system.

## Data

The project is based on the [Amazon Products Dataset 2023](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products).
