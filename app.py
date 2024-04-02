import streamlit as st
import pandas as pd
import pandas as pd
import datetime as dt

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

st.write("Hello World!")

job_df = pd.read_csv('postings.csv')
profile_df = pd.read_csv('updated_profile.csv')

title_list = list(job_df.job_title_categorized.unique())

# Job title dropdown:
with st.sidebar:
    st.write("Please choose a job title")
    title_selector = st.multiselect("Select job title", title_list)

# Word cloud
st.write("Job Description Word Cloud")
jd_by_title = job_df[job_df.job_title_categorized.isin(title_selector)]
job_description_text = ' '.join(jd for jd in jd_by_title[~jd_by_title['Job_description'].isna()]['Job_description'])

# Word cloud 
word_cloud_jd = WordCloud(collocation_threshold = 2, width=1000, height=500,
                        background_color = 'white',
                    ).generate(job_description_text)

# Display the generated Word Cloud
plt.figure( figsize=(10,5) )
plt.imshow(word_cloud_jd)
plt.axis("off")
plt.show()
