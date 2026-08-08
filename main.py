import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def scrape_text_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return ' '.join([p.get_text() for p in paragraphs])

def process_entities(text, entity_types=["GPE", "ORG", "PERSON"]):
    doc = nlp(text)
    return [
        " ".join(ent.text.split())
        for ent in doc.ents
        if ent.label_ in entity_types and len(ent.text.strip()) > 2
    ]

def plot_top_entities(entities, top_n=10):
    counts = Counter(entities).most_common(top_n)
    if not counts:
        print("No matching entities found to plot.")
        return

    df = pd.DataFrame(counts, columns=["Entity", "Frequency"])

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Frequency", y="Entity", hue="Entity", palette="viridis", legend=False)
    plt.title(f"Top {top_n} Most Frequent Entities Scraped from Web")
    plt.xlabel("Frequency Count")
    plt.ylabel("Entity Name")
    plt.tight_layout()
    plt.savefig("web_entity_frequency_plot.png")
    plt.show()

if __name__ == "__main__":
    target_url = "https://en.wikipedia.org/wiki/Malawi"
    
    print(f"Scraping text from: {target_url}...")
    try:
        raw_text = scrape_text_from_url(target_url)
        print(f"Extracted {len(raw_text)} characters of text.")
        
        print("Running Named Entity Recognition...")
        entities = process_entities(raw_text)
        print(f"Extracted {len(entities)} entity mentions.")
        
        plot_top_entities(entities, top_n=10)
    except Exception as e:
        print(f"Error during scraping/processing: {e}")
