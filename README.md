# LinkedIn Job Postings and Profiles Analysis 

**Link to Overview Report can be found [here](https://datalore.jetbrains.com/report/static/0CL9x6jJe9P5sBouyy2SYq/zDL8zEkM5UpHNFbn54AW2H).**

**Abstract:** This project analyzes job postings and LinkedIn profiles in the data field, focusing on insights into the job market and skill requirements. It employs web scraping to gather data, followed by thorough cleaning, exploratory data analysis, and visualization. Using Natural Language Processing, relevant skills are extracted from job descriptions to create a recommendation system for matching job postings with candidate profiles. Additionally, a Random Forest Classifier model predicts job prospects based on education, skills, and experience, evaluated through various metrics and feature importance analysis. The project aims to offer valuable insights for job seekers, employers, and educational institutions in data science and analytics. 

## Data sets

### postings.csv

This file contains the job postings data extracted from various sources. The columns in this file are:

| Column | Description |
| --- | --- |
| industries | The industry the job belongs to |
| city | The city where the job is located |
| state | The state where the job is located |
| job_title_categorized | The categorized job title (e.g., data scientist, data analyst, etc.) |
| job_title | The original job title from the posting |
| job_link | The URL link to the job posting |
| company | The company offering the job |
| company_link | The URL link to the company's website |
| post_time | The date and time when the job was posted |
| applicants_count | The number of applicants for the job (if available) |
| job_description | The full job description text |
| seniority_level | The seniority level of the job (e.g., entry-level, mid-level, senior, etc.) |
| employment_type | The employment type (e.g., full-time, part-time, contract, etc.) |
| job_function | The job function or department (e.g., data science, analytics, engineering, etc.) |

### updated_profile.csv

This file contains the processed LinkedIn profile data. The columns in this file are:

| Column | Description |
| --- | --- |
| User name | The name of the LinkedIn user |
| Headline | The headline or current role mentioned in the user's profile |
| About | The "About" section content from the user's profile |
| Job_title | The categorized job title based on the user's profile summary |
| Experience | The user's work experience summary |
| Company | The user's current or most recent company |
| Company_size | The categorized size of the user's company (Big Tech, Other, or Grad School) |
| University | The user's university or college |
| Degree | The user's degree (e.g., Bachelor's, Master's, PhD) |
| Degree_type | The categorized degree type (Bachelor, Master, PhD, High School, or Other) |
| Major | The user's major or field of study |
| Python | Whether the user has Python skills (yes/no) |
| Java | Whether the user has Java skills (yes/no) |
| SQL | Whether the user has SQL skills (yes/no) |
| Machine_learning | Whether the user has machine learning skills (yes/no) |
| Statistical_analysis | Whether the user has statistical analysis skills (yes/no) |
| Visualization | Whether the user has data visualization skills (yes/no) |
| Software_development | Whether the user has software development skills (yes/no) |
| Git | Whether the user has Git skills (yes/no) |
| HTML_CSS | Whether the user has HTML/CSS skills (yes/no) |
| AI | Whether the user has artificial intelligence skills (yes/no) |
| Has_certification | Whether the user has any certifications (1 for yes, 0 for no) |
| Follower_count | The user's number of followers on LinkedIn |
| Connections | The user's number of connections on LinkedIn |
| Has_job | Whether the user currently has a job (1 for yes, 0 for no) |
| Uni_ranking | The national ranking of the user's university |

## Project Structure

The project is organized into the following main components:

1. **Data Collection and Preprocessing**: Scripts and notebooks for web scraping job postings and LinkedIn profiles, as well as cleaning and preprocessing the collected data.

2. **Exploratory Data Analysis (EDA)**: Notebooks and scripts for exploratory data analysis, including visualizations and insights into the job market and skill requirements.

3. **Natural Language Processing (NLP)**: Notebooks and scripts for applying NLP techniques to analyze job descriptions and extract relevant skills.

4. **Job Recommendation System**: Code for implementing a recommendation system that matches job postings with candidate profiles based on skills and experience.

5. **Predictive Modeling**: Notebooks and scripts for building and evaluating a Random Forest Classifier model to predict the likelihood of securing a data job based on an individual's profile.

6. **Data Files**: The necessary data files, including job postings (`postings.csv`), LinkedIn profiles (`updated_profile.csv`), university rankings, and other relevant data sources.

7. **Visualizations**: Generated visualizations, such as bar charts, pie charts, word clouds, and feature importance plots.

8. **Documentation**: This README file and any additional documentation or reports related to the project.
