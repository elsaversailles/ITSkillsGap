import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy's medium English model
nlp = spacy.load("en_core_web_md")

# Load the dataset
file_path = "top10_PH_IT_Comp_GPC.txt"  # Replace with the actual path
with open(file_path, 'r', encoding='utf-8') as file:
    dataset = file.readlines()

# Define category headers
category_headers = [
    "1. Tech Stack Requirements:",
    "2. IT-Related Qualifications & Skills:",
    "3. Years of Experience:",
    "4. Non-IT Skills Required:",
    "5. Key Responsibilities:"
]

# Process text with spaCy
def preprocess_text_spacy(text):
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(words)

# Precompute embeddings for each category
def compute_embeddings_by_category(dataset, category_headers):
    category_texts = {header: [] for header in category_headers}
    current_category = None

    for line in dataset:
        line_strip = line.strip()
        # Detect category headers
        if any(header.lower() in line_strip.lower() for header in category_headers):
            current_category = next(header for header in category_headers if header.lower() in line_strip.lower())
        elif current_category:
            # Add lines to the current category
            category_texts[current_category].append(line_strip)

    # Compute embeddings
    embeddings = {}
    preprocessed_texts = {}

    for category, lines in category_texts.items():
        combined_text = " ".join(lines)
        preprocessed_text = preprocess_text_spacy(combined_text)
        if preprocessed_text:  # Only process non-empty content
            preprocessed_texts[category] = preprocessed_text
            embeddings[category] = nlp(preprocessed_text).vector

    return preprocessed_texts, embeddings

# Compute embeddings for the dataset
preprocessed_texts, embeddings = compute_embeddings_by_category(dataset, category_headers)

# Debugging: Check embeddings
print("Debug: Total Categories Processed:", len(preprocessed_texts))
for category, text in preprocessed_texts.items():
    print(f"Category: {category}\nPreprocessed Text: {text[:100]}...")
if not embeddings:
    raise ValueError("No embeddings were generated. Check the dataset and preprocessing steps.")

# Function for vector search
def vector_search_by_category(query, preprocessed_texts, embeddings, top_n=3):
    query_vector = nlp(preprocess_text_spacy(query)).vector.reshape(1, -1)  # Convert query to vector and reshape
    results = []

    for category, category_embedding in embeddings.items():
        category_embedding_reshaped = category_embedding.reshape(1, -1)
        similarity = cosine_similarity(query_vector, category_embedding_reshaped)[0][0]
        results.append((category, similarity))

    # Sort by similarity, descending
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:top_n]

# Example query
query = "Develop scalable applications with Python and AWS."
results = vector_search_by_category(query, preprocessed_texts, embeddings)

# Display results
print("Query:", query)
print("\nTop Matches by Category:")
for category, score in results:
    print(f"{category} (Similarity: {score:.4f})")
