USE environmental_policy_db;
SET GLOBAL local_infile = 1;

-- 1. Load Documents Export
LOAD DATA LOCAL INFILE 'C:/Users/User/OneDrive/Desktop/malawi_nlp_pipeline/data/mysql_documents_export.csv'
INTO TABLE documents
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(doc_id, title, source_url, dominant_topic);

-- 2. Create Staging Table and Load Entity Data
CREATE TEMPORARY TABLE raw_entities_stage (
    doc_id INT,
    entity_text VARCHAR(255),
    entity_label VARCHAR(50)
);

LOAD DATA LOCAL INFILE 'C:/Users/User/OneDrive/Desktop/malawi_nlp_pipeline/data/mysql_entities_export.csv'
INTO TABLE raw_entities_stage
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(doc_id, entity_text, entity_label);

-- 3. Populate Entities and Junction Table
INSERT IGNORE INTO entities (entity_text, entity_label)
SELECT DISTINCT entity_text, entity_label 
FROM raw_entities_stage;

INSERT INTO document_entities (doc_id, entity_id, mention_count)
SELECT 
    s.doc_id,
    e.entity_id,
    COUNT(*) as mention_count
FROM raw_entities_stage s
JOIN entities e ON s.entity_text = e.entity_text AND s.entity_label = e.entity_label
GROUP BY s.doc_id, e.entity_id;

DROP TEMPORARY TABLE raw_entities_stage;

-- 4. Create Policy Analytics View
CREATE OR REPLACE VIEW vw_document_policy_summary AS
WITH DocEntityStats AS (
    SELECT 
        doc_id,
        COUNT(DISTINCT entity_id) AS total_unique_entities,
        SUM(mention_count) AS total_entity_mentions
    FROM document_entities
    GROUP BY doc_id
)
SELECT 
    d.doc_id,
    d.title AS document_title,
    t.topic_label AS primary_theme,
    COALESCE(s.total_unique_entities, 0) AS unique_entity_count,
    COALESCE(s.total_entity_mentions, 0) AS total_entity_mentions,
    ROUND(
        COALESCE(s.total_entity_mentions, 0) / NULLIF(s.total_unique_entities, 0), 2
    ) AS entity_repetition_ratio
FROM documents d
LEFT JOIN topics t ON d.dominant_topic = t.topic_id
LEFT JOIN DocEntityStats s ON d.doc_id = s.doc_id;

-- 5. Key Actor Policy Share Analysis
SELECT 
    e.entity_text AS key_actor,
    e.entity_label AS category,
    t.topic_label AS policy_domain,
    SUM(de.mention_count) AS mentions_in_topic,
    totals.global_entity_mentions,
    totals.total_docs_spanned,
    ROUND((SUM(de.mention_count) / totals.global_entity_mentions) * 100, 2) AS topic_share_percentage
FROM document_entities de
JOIN entities e ON de.entity_id = e.entity_id
JOIN documents d ON de.doc_id = d.doc_id
JOIN topics t ON d.dominant_topic = t.topic_id
JOIN (
    SELECT 
        entity_id,
        SUM(mention_count) AS global_entity_mentions,
        COUNT(DISTINCT doc_id) AS total_docs_spanned
    FROM document_entities
    GROUP BY entity_id
) totals ON e.entity_id = totals.entity_id
WHERE e.entity_label IN ('ORG', 'GPE')
GROUP BY e.entity_id, e.entity_text, e.entity_label, t.topic_label, totals.global_entity_mentions, totals.total_docs_spanned
ORDER BY totals.global_entity_mentions DESC, key_actor, topic_share_percentage DESC;