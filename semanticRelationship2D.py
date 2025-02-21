#No Semantic Chunking

import os
import re
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import nltk
from plotly.subplots import make_subplots
import pandas as pd
import time

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocesses text by lowercasing, removing non-alphanumeric characters, tokenizing, removing stopwords, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def chunk_text(text, chunk_size=500):
    """Chunks text into smaller segments of approximately chunk_size words."""
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def compare_documents(course_text, listings_text, model):
    """Compares course and listings documents by calculating cosine similarity and preparing data for visualization."""
    course_text = preprocess_text(course_text)
    listings_text = preprocess_text(listings_text)

    if not course_text:  # Handle empty course text after preprocessing
        print("Course text is empty after preprocessing. Skipping comparison.")
        return None, None

    if not listings_text: # Handle empty listings text after preprocessing
        print("Listings text is empty after preprocessing. Skipping comparison.")
        return None, None

    course_embedding = model.encode(course_text, convert_to_tensor=True).cpu().numpy()
    listings_embedding = model.encode(listings_text, convert_to_tensor=True).cpu().numpy()

    cosine_sim = util.cos_sim(course_embedding, listings_embedding).item()

    return cosine_sim, course_embedding

def create_combined_visualization(all_data, listings_embeddings, output_html="combined_semantic_Fixed_length-chunking_2Drelationships.html"):
    """Creates a combined visualization of semantic relationships for all courses and listings."""
    all_embeddings = np.vstack([embedding for _, embedding in all_data] + [embedding for embedding in listings_embeddings])
    n_samples = all_embeddings.shape[0]
    perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than the number of samples

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    labels = [name for name, _ in all_data] + [f'Listings Chunk {i+1}' for i in range(len(listings_embeddings))]
    colors = ['Course'] * len(all_data) + ['Listings'] * len(listings_embeddings)

    df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "Label": labels,
        "Document": colors
    })

    fig = px.scatter(df, x="x", y="y", color="Document", hover_data=["Label"])
    fig.update_traces(text=None)  # Remove persistent text labels
    fig.update_layout(
        height=600,
        title_text="Semantic Similarity with Top 10 PH Companies (Fixed Chunking)",
        xaxis=dict(range=[-30, 30]),  # Set x-axis range
        yaxis=dict(range=[-30, 30])   # Set y-axis range
    )
    fig.write_html(output_html)
    print(f"Combined visualization saved to {output_html}")


if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    listings_file = "/workspaces/ITSkillsGap/Skill/top10_PH_Comp_Market_cap.txt"  # Replace with your listings file path
    course_dir = "/workspaces/ITSkillsGap/Skill/Docs/Course_Itemized"  # Replace with your course directory

    try:
        with open(listings_file, 'r') as file:
            listings_text = file.read()
    except FileNotFoundError:
        print(f"Error: Listings file not found at {listings_file}")
        exit()

    model = SentenceTransformer('all-mpnet-base-v2')
    listings_chunks = chunk_text(listings_text)
    listings_embeddings = model.encode(listings_chunks, convert_to_tensor=True).cpu().numpy()

    all_data = []
    for filename in os.listdir(course_dir):
        if filename.endswith(".txt"):
            course_file = os.path.join(course_dir, filename)
            course_name = filename[:-4]

            try:
                with open(course_file, 'r') as file:
                    course_text = file.read()
            except FileNotFoundError:
                print(f"Error: Course file not found at {course_file}. Skipping.")
                continue  # Skip to the next file

            similarity_score, course_embedding = compare_documents(course_text, listings_text, model)

            if similarity_score is not None:  # Only add if comparison was successful
                all_data.append((course_name, course_embedding))
                print(f"Cosine Similarity between {course_name} and Listings: {similarity_score:.5f}")

    if all_data:  # Check if any course data was successfully processed
        create_combined_visualization(all_data, listings_embeddings)
    else:
        print("No course data was successfully processed. Visualization not created.")


    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Execution time: {execution_time:.2f} seconds")
