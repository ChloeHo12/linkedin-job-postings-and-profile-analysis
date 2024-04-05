import streamlit as st
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# NLP library
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics.pairwise import linear_kernel

# Page title
st.set_page_config(
    page_title="Navigating Data Job and Skills", #title shown in browser tab
    #page_icon="✅", #emoji as string or in shortcode or pass URL/np array of image
    layout = "wide"
)
st.markdown("<h1 style='text-align: center;'>Data Skills Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Unveiling insights into the job market and skill requirements</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Public data scraped from LinkedIn", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)  # Add line separator

# Load dataframes
job_df = pd.read_csv("postings.csv")
profile_df = pd.read_csv("updated_profile.csv")

title_list = list(job_df[job_df['job_title_categorized'] != 'other']['job_title_categorized'].unique())

# Metrics
number_of_jobpostings = job_df.shape[0]
number_of_cities = job_df['City'].nunique()
number_of_profiles = profile_df.shape[0]
with st.sidebar:
    # Job title dropdown
    st.write("Please choose a job title")
    title_selector = st.selectbox("Selected job title", title_list)
    st.write("") 
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)  # Add line separator
    st.metric(
        label='Number of Job Postings',
        value=int(number_of_jobpostings),
    )
    st.metric(
        label='Number of Cities',
        value=int(number_of_cities),
    )
    st.metric(
        label='Number of Profiles',
        value=int(number_of_profiles),
    )

# EDA
# Distribution of job titles
st.markdown(f"<h3 style='color:#B19CD9;'>Exploratory Data Analysis</h3>", unsafe_allow_html=True)
job_counts = job_df['job_title_categorized'].value_counts()
job_counts_df = pd.DataFrame(job_counts).reset_index()
job_counts_df.columns = ['job_title_categorized', 'count']
fig = px.pie(job_counts_df, values='count', names='job_title_categorized', title='Distribution of job titles', color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(width=500, height=400)  
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig)

# Distribution of industries
industry_counts = job_df['Job_function'].value_counts()
threshold = 20
other_count = industry_counts[industry_counts < threshold].sum()
industry_counts_filtered = industry_counts[industry_counts >= threshold]
industry_counts_df = pd.DataFrame(industry_counts_filtered).reset_index()
industry_counts_df.columns = ['Job_function', 'count']
fig2 = px.pie(industry_counts_df, values='count', names='Job_function', title='Distribution of industries')
fig2.update_layout(width=600, height=400)  
with col2:
    st.plotly_chart(fig2)

# Top Skills Extract
selected_titles = [title_selector]
jd_by_title = job_df[job_df.job_title_categorized.isin(selected_titles)]
job_description_text = ' '.join(jd for jd in jd_by_title[~jd_by_title['Job_description'].isna()]['Job_description'])

relevant_skills = ['python', 'r', 'sql', 'a/b', 'etl', 'experiment', 'modeling', 'pipeline', 'roadmap', 'critical thinking', 
                   'statistics', 'machine learning', 'data visualization', 'communication', 'user experience'
                   'problem solving', 'mathematics', 'natural language processing', 'causal inference', 'strategy'
                   'deep learning', 'spark', 'tableau', 'powerbi', 'java', 'oop', 'hadoop', 'git', 'html', 'css',
                   'product analysis']
tokens = word_tokenize(job_description_text.lower())
stop_words = set(stopwords.words('english'))

filtered_skills= [token for token in tokens if token.isalnum() and token not in stop_words and token.lower() in relevant_skills]
cleaned_text_skills = ' '.join(filtered_skills)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([cleaned_text_skills])
skill_names = vectorizer.get_feature_names_out()
tfidf_scores = X.T.toarray().flatten()
skill_tfidf_scores = list(zip(skill_names, tfidf_scores))
sorted_skills = sorted(skill_tfidf_scores, key=lambda x: x[1], reverse=True)

# Print top 10 most important skills
top_skills = [skill for skill, _ in sorted_skills[:10]]
word_counts = Counter(filtered_skills)
fig, ax = plt.subplots(figsize=(18, 14))
words, counts = zip(*word_counts.most_common(10))  # Get top 10 words and their counts
ax.bar(words, counts, color='lightsteelblue')
ax.set_title('Top 10 Skills required for ' + title_selector, fontsize=25)
ax.set_xlabel('Words', fontsize=20)
ax.set_ylabel('Frequency', fontsize=20)
plt.xticks(rotation=45, fontsize=20)  # Rotate x-labels for better readability
plt.tight_layout()

# Top libraries extract
relevant_libraries = ['matplotlib', 'numpy', 'pandas', 'nltk', 'seaborn', 'scikit-learn', 'plotly', 'tensorflow',
                     'pytorch', 'keras', 'beautifulsoup', 'scipy', 'statsmodels']
filtered_lib = [token for token in tokens if token.isalnum() and token not in stop_words and token.lower() in relevant_libraries]
cleaned_text_lib = ' '.join(filtered_lib)
X = vectorizer.fit_transform([cleaned_text_lib])
lib_names = vectorizer.get_feature_names_out()
tfidf_scores = X.T.toarray().flatten()
lib_tfidf_scores = list(zip(lib_names, tfidf_scores))
sorted_lib = sorted(lib_tfidf_scores, key=lambda x: x[1], reverse=True)

# Print top 10 most important libs
top_libs = [lib for lib, _ in sorted_lib[:10]]
word_counts_lib = Counter(filtered_lib)
fig2, ax = plt.subplots(figsize=(18, 14))
words_lib, counts_lib = zip(*word_counts_lib.most_common(10))  # Get top 10 words and their counts
ax.bar(words_lib, counts_lib, color = 'plum')
ax.set_title('Top 10 common libraries used by ' + title_selector, fontsize=25)
ax.set_xlabel('Words', fontsize=20)
ax.set_ylabel('Frequency', fontsize=20)
plt.xticks(rotation=45, fontsize=20)  # Rotate x-labels for better readability
plt.tight_layout()

st.markdown(f"<h3 style='color:#B19CD9;'>Top Skills and Libraries for {title_selector}</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig)
with col2:
    st.pyplot(fig2)

# Recommendation system
st.write("") 
st.write("") 
st.markdown(f"<h3 style='color:#B19CD9;'>Recommended jobs for user profile</h3>", unsafe_allow_html=True)
user_profile = st.text_input("Enter your profile: (i.e. I am a data scientist with experience in machine learning, Python, and SQL. I am interested in roles related to predictive modeling, and developing AI solutions.)"
)  
job_data = job_df

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(job_data['Job_description'].values.astype('U'))

def get_job_recommendations(user_profile, top_n):
    user_profile_vec = tfidf.transform([user_profile])
    cosine_similarities = linear_kernel(user_profile_vec, tfidf_matrix)
    top_indices = cosine_similarities.argsort()[0][-top_n:][::-1]
    top_jobs = job_data.iloc[top_indices][['Job_title', 'Company', 'Job_link', 'Job_description']]
    unique_companies = set()
    filtered_jobs = []
    
    for index, job in top_jobs.iterrows():
        if job['Company'] not in unique_companies:
            filtered_jobs.append(job)
            unique_companies.add(job['Company'])
            if len(filtered_jobs) == top_n:
                break
    
    filtered_jobs_df = pd.DataFrame(filtered_jobs)
    
    return filtered_jobs_df

recommended_jobs = get_job_recommendations(user_profile, top_n=10)
st.write(recommended_jobs)
