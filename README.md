# LinkedIn Job Postings and Profiles Analysis 

**Link to Overview Report can be found [here](https://datalore.jetbrains.com/report/static/0CL9x6jJe9P5sBouyy2SYq/zDL8zEkM5UpHNFbn54AW2H).**

**Abstract:** This project analyzes job postings and LinkedIn profiles in the data field, focusing on insights into the job market and skill requirements. It employs web scraping to gather data, followed by thorough cleaning, exploratory data analysis, and visualization. Using Natural Language Processing, relevant skills are extracted from job descriptions to create a recommendation system for matching job postings with candidate profiles. Additionally, a Random Forest Classifier model predicts job prospects based on education, skills, and experience, evaluated through various metrics and feature importance analysis. The project aims to offer valuable insights for job seekers, employers, and educational institutions in data science and analytics. 

## Data Collection

The project starts by web scraping job postings and LinkedIn profiles from different cities using the LinkedIn website. The scraped data is stored in separate CSV files for each city, including: Seattle, New York, Austin, Chicago, Boston, San Francisco, Richmond, Baltimore. The two data sets can be found below:

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

## Data Cleaning and EDA

The data cleaning process involves the following steps:

1. **Loading Data**: The job posting data is loaded from the individual city CSV files.
2. **Sampling**: For the analysis, only the first 430 rows from each city's dataset are retained.
3. **Concatenation**: The sampled data from all cities is concatenated into a single DataFrame named `job_df`.
4. **Column Manipulation**: Unnecessary columns are dropped, and the location information is split into separate 'City' and 'State' columns.
5. **Job Title Categorization**: A new column 'job_title_categorized' is created, which categorizes job titles based on keywords like 'scientist', 'analyst', 'engineer', etc.
6. **Data Exploration**: The distribution of job titles and industries is visualized using bar charts and pie charts.
7. **Word Cloud**: A word cloud is generated for the job descriptions of a selected job title (e.g., 'data analyst') to understand the commonly used words and skills required.

## NLP Analysis

The project utilizes natural language processing (NLP) techniques to analyze job descriptions and extract relevant skills. The main steps include:

1. **Tokenization**: The job descriptions are tokenized using NLTK.
2. **Skill Filtering**: A list of relevant skills is defined, and the tokenized job descriptions are filtered to keep only the relevant skills.
3. **TF-IDF Vectorization**: The filtered job descriptions are vectorized using TF-IDF to calculate the importance of each skill.
4. **Top Skills**: The top 10 most important skills are identified based on their TF-IDF scores and visualized using a bar chart.

**A demo can be found here:**

https://github.com/ChloeHo12/linkedin-job-postings-and-profile-analysis/assets/98048503/e4b2a56e-5453-480a-8939-a33cebb5f3e1



## Recommendation System

A recommendation system is implemented to match job postings with a candidate's profile. The steps involved are:

1. **Data Preparation**: The job postings data is loaded from the 'postings.csv' file.
2. **TF-IDF Matrix**: A TF-IDF matrix is created from the job descriptions.
3. **Cosine Similarity**: The cosine similarity between a given user profile and the job descriptions is calculated using the TF-IDF matrix.
4. **Top Recommendations**: The top N most similar job postings are recommended based on the cosine similarity scores, ensuring that only one job from each unique company is included.

## Predictive Modeling

The project builds a predictive model to estimate the probability of landing a data job based on a candidate's profile. The steps involved are:

1. **Data Preprocessing**: The LinkedIn profile data is loaded and preprocessed, including feature engineering, one-hot encoding of categorical variables, and imputation of missing values.
2. **Feature Selection**: Relevant features are selected based on their importance scores.
3. **Model Training**: A Random Forest Classifier is trained on the preprocessed data.
4. **Model Evaluation**: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
5. **Feature Importance**: The importance of each feature is calculated and visualized.
6. **Optimized Model**: An optimized Random Forest model is built using the selected important features.

## 4. Implications 

**4.1. Implications for Stakeholders:**

- Job Seekers: The project provides valuable insights into job requirements, skill sets, and the probability of landing a data job based on a candidate's profile. This information can help job seekers tailor their resumes, acquire relevant skills, and apply to suitable job opportunities.
- Employers: The project's analysis of job descriptions and requirements can assist employers in crafting more effective job postings and aligning their expectations with industry standards.
- Educational Institutions: The identification of in-demand skills can help educational institutions update their curricula and prepare students for the job market.
- Career Counselors: The project's findings can aid career counselors in providing better guidance to individuals interested in data-related fields.

**4.2. Ethical, Legal, and Societal Implications:**

- Ethical Considerations: The project scrapes public LinkedIn profiles, the project does not violate any terms of service or privacy policies.
- Legal Implications: The project's findings can contribute to the development of unbiased hiring policies, promoting equal employment opportunities and preventing discrimination.
- Societal Impact: The project empowers job seekers by providing valuable insights into the skills and qualifications required in the data science and analytics fields, potentially leading to better career opportunities.




