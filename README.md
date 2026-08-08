# Environmental Policy & Carbon Governance NLP Analytics Pipeline

An end-to-end computational text processing and relational database pipeline built to analyze environmental policy documents, actor networks, and carbon governance frameworks in Malawi.

## Project Overview
This repository integrates Natural Language Processing (NLP), relational database engineering, and network visualization to track key institutional actors, policy themes, and entity co-occurrences across environmental governance texts.

## Architecture & Workflow
1. **Text Processing & NER (`scripts/advanced_nlp_pipeline.py`)**: Processes raw document corpora through `spaCy` to extract Named Entities (`ORG`, `GPE`) and `scikit-learn` Latent Dirichlet Allocation (LDA) for thematic topic modeling.
2. **Relational ETL & Analytics (`scripts/database_schema_and_queries.sql`)**: Ingests structured entity/document exports into a normalized MySQL schema (`environmental_policy_db`), leveraging temporary staging tables, duplicate handling, and CTE-based analytical views (`vw_document_policy_summary`).
3. **Data Outputs & Visualizations (`data/` & `visuals/`)**: Stores co-occurrence network edge lists, key actor topic-share calculations, and exported frequency plots.

## Repository Structure
- `data/`: Ingestion CSV exports and SQL analytics output datasets.
- `scripts/`: Python NLP pipelines, database DDL schema, and analytical SQL scripts.
- `visuals/`: Generated entity co-occurrence and frequency network plots.

## Tech Stack
- **Languages**: Python, SQL, R
- **Libraries**: `spaCy`, `scikit-learn`, `pandas`, `matplotlib`
- **Database**: MySQL 8.0 / MySQL Workbench
- **Version Control**: Git / GitHub# Carbon & Environmental Policy NLP Tracker`n`nAn automated Python pipeline that scrapes web data, executes Named Entity Recognition (NER) using spaCy, and visualizes entity frequency distributions.`n`n## Pipeline Visual Output`n`n![Web Entity Frequency Plot](web_entity_frequency_plot.png)`n`n## How to Run`n`n```cmd`npip install requests beautifulsoup4 spacy pandas seaborn matplotlib`npython -m spacy download en_core_web_sm`npython main.py`n```
