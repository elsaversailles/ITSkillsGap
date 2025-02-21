import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import nltk

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

def extract_skills(text):
    """Extracts skills and qualifications from text using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    skill_scores = dict(zip(feature_names, scores))
    return skill_scores

def identify_skill_gaps(job_postings, curriculum):
    """Identifies skill gaps between job postings and curriculum."""
    job_postings_text = ' '.join(job_postings)
    curriculum_text = ' '.join(curriculum)
    
    job_postings_skills = extract_skills(preprocess_text(job_postings_text))
    curriculum_skills = extract_skills(preprocess_text(curriculum_text))
    
    skill_gaps = {skill: job_postings_skills[skill] for skill in job_postings_skills if skill not in curriculum_skills}
    
    return skill_gaps

if __name__ == "__main__":
    listings_file = "/workspaces/ITSkillsGap/Skill/top10_PH_Comp_Market_cap.txt"  # Replace with your listings file path
    course_dir = "/workspaces/ITSkillsGap/Skill/Docs/Course_Itemized"  # Replace with your course directory

    try:
        with open(listings_file, 'r') as file:
            listings_text = file.read()
    except FileNotFoundError:
        print(f"Error: Listings file not found at {listings_file}")
        exit()

    job_postings = [listings_text]

    curriculum = []
    for filename in os.listdir(course_dir):
        if filename.endswith(".txt"):
            course_file = os.path.join(course_dir, filename)
            try:
                with open(course_file, 'r') as file:
                    curriculum.append(file.read())
            except FileNotFoundError:
                print(f"Error: Course file not found at {course_file}. Skipping.")
                continue  # Skip to the next file

    skill_gaps = identify_skill_gaps(job_postings, curriculum)
    
    # Save the results to a CSV file
    output_file = "/workspaces/ITSkillsGap/phComp.csv"
    skill_gaps_df = pd.DataFrame(skill_gaps.items(), columns=['Skill', 'Score'])
    skill_gaps_df.to_csv(output_file, index=False)

    print(f"Identified skill gaps have been saved to {output_file}")
