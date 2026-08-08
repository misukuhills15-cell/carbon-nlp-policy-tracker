import itertools
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import spacy
from bs4 import BeautifulSoup
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ==========================================
# 1. DATA ACQUISITION & PARSING
# ==========================================
URLS = [
    "https://en.wikipedia.org/wiki/Carbon_footprint",
    "https://en.wikipedia.org/wiki/Climate_change_policy_of_the_United_States",
    "https://en.wikipedia.org/wiki/Kyoto_Protocol",
]

# Fallback text in case external web requests fail or get blocked
FALLBACK_TEXTS = {
    1: "Carbon footprint measurements evaluate total greenhouse gas emissions caused directly and indirectly by an individual, organization, event, or product. The United Nations Framework Convention on Climate Change (UNFCCC) tracks global carbon metrics, carbon trading mechanisms, and Article 6 carbon offset frameworks.",
    2: "Climate change policy of the United States involves federal regulations, environmental protection laws, and clean energy incentives. The Environmental Protection Agency (EPA) oversees policy enforcement alongside international treaty commitments like the Paris Agreement.",
    3: "The Kyoto Protocol was an international treaty that extended the 1992 UNFCCC that commits state parties to reduce greenhouse gas emissions. It established market-based mechanisms such as emissions trading, the Clean Development Mechanism (CDM), and Joint Implementation."
}

def fetch_and_clean_text(urls):
    """Scrapes raw paragraph text from target URLs using BeautifulSoup with fallback support."""
    documents = []
    print("[+] Fetching and scraping policy text from target URLs...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    for doc_id, url in enumerate(urls, start=1):
        clean_text = ""
        title = f"Doc_{doc_id}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                paragraphs = soup.find_all("p")
                full_text = " ".join([p.get_text() for p in paragraphs])
                clean_text = re.sub(r"\[\d+\]", "", full_text)
                clean_text = re.sub(r"\s+", " ", clean_text).strip()
                if soup.title and soup.title.string:
                    title = soup.title.string
        except Exception as e:
            print(f"[!] Warning: Request failed for {url} ({e}). Using fallback text.")

        # Use fallback text if scraped text is empty or too short
        if not clean_text or len(clean_text) < 50:
            print(f"    - Notice: Using fallback policy text for Doc ID {doc_id}.")
            clean_text = FALLBACK_TEXTS.get(doc_id, "Environmental policy and carbon market emissions framework data.")

        documents.append(
            {
                "doc_id": doc_id,
                "title": title,
                "source_url": url,
                "raw_text": clean_text,
            }
        )
        print(f"    - Loaded Doc ID {doc_id}: {len(clean_text):,} characters.")

    # Explicitly enforce DataFrame columns
    df = pd.DataFrame(documents, columns=["doc_id", "title", "source_url", "raw_text"])
    return df


# ==========================================
# 2. SPACY NER & CO-OCCURRENCE EXTRACTION
# ==========================================
def extract_ner_and_cooccurrence(
    df, nlp, target_labels=["ORG", "GPE", "PERSON", "NORP", "EVENT"]
):
    """Extracts Named Entities and builds sentence-level entity co-occurrence edges."""
    print("\n[+] Processing texts through spaCy NER pipeline...")
    extracted_entities = []
    co_occurrence_pairs = []

    for idx, row in df.iterrows():
        doc_id = row["doc_id"]
        spacy_doc = nlp(row["raw_text"])

        for sent in spacy_doc.sents:
            sent_entities = list(
                set(
                    [
                        ent.text.strip()
                        for ent in sent.ents
                        if ent.label_ in target_labels and len(ent.text.strip()) > 2
                    ]
                )
            )

            for ent in sent.ents:
                if ent.label_ in target_labels and len(ent.text.strip()) > 2:
                    extracted_entities.append(
                        {
                            "doc_id": doc_id,
                            "entity_text": ent.text.strip(),
                            "entity_label": ent.label_,
                        }
                    )

            if len(sent_entities) >= 2:
                for source, target in itertools.combinations(
                    sorted(sent_entities), 2
                ):
                    co_occurrence_pairs.append((source, target))

    if extracted_entities:
        df_entities = pd.DataFrame(extracted_entities)
    else:
        df_entities = pd.DataFrame(columns=["doc_id", "entity_text", "entity_label"])

    edge_counts = Counter(co_occurrence_pairs)
    if edge_counts:
        df_edges = pd.DataFrame(
            [
                {
                    "source": pair[0],
                    "target": pair[1],
                    "weight": weight,
                    "relationship": "CO_OCCURS_WITH",
                }
                for pair, weight in edge_counts.items()
            ]
        ).sort_values(by="weight", ascending=False)
    else:
        df_edges = pd.DataFrame(columns=["source", "target", "weight", "relationship"])

    return df_entities, df_edges


# ==========================================
# 3. SCIKIT-LEARN LDA TOPIC MODELING
# ==========================================
def perform_lda_topic_modeling(df, n_topics=3, n_top_words=8):
    """Extracts latent topics across scraped documents using LDA."""
    print("\n[+] Running Latent Dirichlet Allocation (LDA) Topic Modeling...")

    tf_vectorizer = CountVectorizer(
        max_df=0.95, min_df=1, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )
    tf_matrix = tf_vectorizer.fit_transform(df["raw_text"])
    feature_names = tf_vectorizer.get_feature_names_out()

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="online",
        max_iter=20,
    )
    lda_matrix = lda_model.fit_transform(tf_matrix)

    topics_dict = {}
    print("\n--- Discovered Policy Topics ---")
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_words = [feature_names[i] for i in top_features_ind]
        topics_dict[f"Topic_{topic_idx + 1}"] = top_words
        print(f"  • Topic #{topic_idx + 1}: {', '.join(top_words)}")

    df["dominant_topic"] = np.argmax(lda_matrix, axis=1) + 1
    return df, topics_dict


# ==========================================
# 4. MAIN PIPELINE EXECUTION & EXPORT
# ==========================================
def main():
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000

    df_docs = fetch_and_clean_text(URLS)
    df_entities, df_edges = extract_ner_and_cooccurrence(df_docs, nlp)
    df_docs, topics_summary = perform_lda_topic_modeling(df_docs, n_topics=3)

    print("\n[+] Exporting processed pipeline artifacts...")
    df_docs[["doc_id", "title", "source_url", "dominant_topic"]].to_csv(
        "mysql_documents_export.csv", index=False
    )
    df_entities.to_csv("mysql_entities_export.csv", index=False)
    df_edges.to_csv("r_network_edgelist.csv", index=False)

    print("    - Exported: 'mysql_documents_export.csv'")
    print("    - Exported: 'mysql_entities_export.csv'")
    print("    - Exported: 'r_network_edgelist.csv'")

    if not df_edges.empty:
        plt.figure(figsize=(10, 6))
        top_edges = df_edges.head(10).copy()
        top_edges["pair"] = top_edges["source"] + " <-> " + top_edges["target"]

        sns.barplot(data=top_edges, x="weight", y="pair", palette="crest")
        plt.title("Top Entity Co-occurrence Connections (Sentence Level)", fontsize=14)
        plt.xlabel("Co-occurrence Frequency (Shared Sentences)", fontsize=11)
        plt.ylabel("Entity Pair", fontsize=11)
        plt.tight_layout()
        plt.savefig("entity_cooccurrence_plot.png", dpi=300)
        plt.show()
        print("    - Visualization Saved: 'entity_cooccurrence_plot.png'")
    else:
        print("    - Note: No co-occurrence pairs found to plot.")

    print("\n[✓] Advanced NLP Pipeline execution complete!")


if __name__ == "__main__":
    main()
