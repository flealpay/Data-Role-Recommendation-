######################### DATA ROLE RECOMMENDATION #########################

# Work Year
# Experience Level
# Employment Type
# Job Title
# Salary
# Employee Residence
# Remote Ratio
# Company Location
# Company Size
# Salary Level
# Country Fullname
# Continent

# pip install pandas numpy
# pip install country_converter
# pip install country_converter[all]
# pip install pycountry
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import seaborn as sns
import country_converter as coco
cc = coco.CountryConverter()
import pycountry
pycountry.countries

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format', lambda x:'%.5f' % x)


df1 = pd.read_csv('ds_salaries.csv')
df = df1.copy()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe().T)

check_df(df)

#######################  Salary and  Salary was removed ######################
df.drop(['salary_currency', 'salary'], axis=1, inplace=True)
df.head()

#######################  Column name is changed ######################
df.rename(columns={'salary_in_usd': 'salary'}, inplace=True)
df.head()


#######################   The "Part Time (PT)" variable was removed as it was affecting our accurate results. ######################

df["employment_type"].unique()

df.drop(df[df['employment_type'] == 'PT'].index, inplace=True)

df.head()

###################### We organize them through a process of consolidation because we have many similar variables.######################

df = df[~df['job_title'].str.contains('Manager')]

df.loc[df['job_title'].str.contains('Data Analyst'), 'job_title'] = 'Data Analyst'

df.loc[df['job_title'].str.contains('Data Scientist'), 'job_title'] = 'Data Scientist'

df.loc[df['job_title'].str.contains('Data Engineer'), 'job_title'] = 'Data Engineer'

df.loc[df['job_title'].str.contains('Data Science'), 'job_title'] = 'Data Scientist'

df.loc[df['job_title'].str.contains('Machine Learning'), 'job_title'] = 'Machine Learning'

df.loc[df['job_title'].str.contains('ML'), 'job_title'] = 'Machine Learning'

df['job_title'].nunique()

def assign_broader_category(job_title):
    data_scientist = ['Principal Data Scientist', 'Data Scientist', 'Applied Scientist', 'Applied Data Scientist',
                      'Data Science Manager', 'Director of Data Science', 'Lead Data Scientist',
                      'Data Science Lead', 'Data Science Consultant', 'Head of Data Science',
                      'Data Science Engineer', 'Data Science Tech Lead', 'Data Scientist Lead',
                      'Product Data Scientist', 'Staff Data Scientist'
                      ]
    data_engineer = ['Data Modeler', 'Data Strategist', 'Data Engineer', 'Data Architect', 'Data DevOps Engineer',
                     'Head of Data', 'Data Manager', 'Data Operations Engineer', 'Azure Data Engineer',
                     'Big Data Engineer', 'Cloud Database Engineer', 'Marketing Data Engineer', 'Data Lead',
                     'Data Infrastructure Engineer', 'Software Data Engineer', 'Data Specialist',
                     'BI Data Engineer', 'BI Developer', 'Big Data Architect', 'Cloud Data Engineer',
                     'Data Operations Analyst', 'Power BI Developer', 'Principal Data Architect',
                     'Cloud Data Architect', 'Lead Data Engineer', 'Principal Data Engineer']
    #     machine_learning= ['ML Engineer','Machine Learning Engineer',
    #  'Applied Machine Learning Engineer','Machine Learning Researcher','Machine Learning Scientist',
    #  'MLOps Engineer', 'AI Scientist','AI Developer','Applied Machine Learning Scientist',
    #  'AI Programmer','Deep Learning Researcher',  'Machine Learning Infrastructure Engineer',
    #  'Deep Learning Engineer', 'Machine Learning Software Engineer','Machine Learning Research Engineer',
    #  'NLP Engineer','Machine Learning Developer', 'Principal Machine Learning Engineer',
    #  'Machine Learning Manager', 'Lead Machine Learning Engineer','Head of Machine Learning',
    # ]
    data_analyst = ['Data Analyst', 'Analytics Engineer', 'Data Quality Analyst', 'Compliance Data Analyst',
                    'Data Analytics Manager', 'Business Data Analyst', 'Staff Data Analyst',
                    'Lead Data Analyst', 'Financial Data Analyst', 'BI Analyst', 'Product Data Analyst',
                    'Data Analytics Lead', 'Data Analytics Specialist', 'BI Data Analyst', 'Insight Analyst',
                    'Data Analytics Engineer', 'Data Analytics Consultant', 'Marketing Data Analyst',
                    'Principal Data Analyst', 'Finance Data Analyst']

    if job_title in data_engineer:
        return "Data Engineer"
    elif job_title in data_scientist:
        return "Data Scientist"
        #     elif job_title in machine_learning:
        #         return "ML Engineer"
    elif job_title in data_analyst:
        return "Data Analyst"
    else:
        return "Machine-learning & Other"

df['job_category'] = df['job_title'].apply(assign_broader_category)

temp = df.groupby(['experience_level', 'job_category']).size().reset_index()
temp.columns = ['experience_level', 'job_category', 'value']
plt.figure(figsize=(10, 5))
ax = sns.barplot(data=temp, x=temp['job_category'], y=temp['value'],
                     hue=temp['experience_level'])
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.1f')
plt.tight_layout()
plt.show()

###################### We look at the average salaries of our variables by year and their titles.######################
maas = df.groupby(["job_title", "work_year"]).agg({"salary": "mean"}).sort_values(by=["work_year", "salary"], ascending=[False, False]) #  Sort in descending order

print(maas)


###################### Displaying the first 10 professions in a table ######################
# The results show that our leading fields are Data Engineer, Data Scientist, Data Analyst, and Machine Learning.

title_counts = df["job_title"].value_counts()

top_10_titles = title_counts.head(10)

top_10_df = pd.DataFrame({'Job Title': top_10_titles.index, 'Count': top_10_titles.values})

# Plotting the count plot
plt.figure(figsize=(12, 6))
sns.set(style="darkgrid")
ax = sns.barplot(data=top_10_df, x='Count', y='Job Title', palette='cubehelix')
plt.xlabel('Count')
plt.ylabel('Job Titles')
plt.title('Top 10 Most Frequent Job Titles')

for i, v in enumerate(top_10_df['Count']):
    ax.text(v + 0.2, i, str(v), color='black', va='center')

plt.tight_layout()
plt.show()


###################### We compare based on experience levels. ######################
# The conclusion we draw from here is that we have many competitors in both the Expert and Intermediate levels, indicating that everyone is continuously improving themselves in this field

level_counts = df['experience_level'].value_counts()
df = pd.DataFrame(df)

df['experience_level'] = df['experience_level'].replace({'SE': 'Expert', 'MI': 'Intermediate', 'EN': 'Junior', 'EX': 'Director'})

df['employment_type'] = df['employment_type'].replace({'FT': 'Full-time', 'CT': 'Contract', 'FL': 'Freelance'})

plt.figure(figsize=(7,12),dpi=80)
plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
plt.title('Experience Level Distribution')

plt.show()

###################### Comparison between company size and experience levels ######################
#### As a result, we actually see a clustering mainly in mid-sized companies, and here we can also say that the working areas in small and large-sized companies involve fewer people ###

cross_tab = pd.crosstab(df['experience_level'], df['company_size'])

plt.figure(figsize=(12, 10))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap='Blues')

plt.xlabel('Company Size')
plt.ylabel('Experience Level')
plt.title('Relationship between Experience Level and Company Size')

plt.show()

###################### Experience Levels by Employment Type  ######################
#### As a result, we observe that our competitors predominantly offer services in the Full-time employment category. When considering Freelance and Contractual employment ####
cross_tab = pd.crosstab(df['experience_level'], df['employment_type'])


plt.figure(figsize=(12, 10))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap='Blues')

plt.xlabel('Employment_type')
plt.ylabel('Experience Level')
plt.title('Relationship between Experience Level and Company Size')

plt.show()


###################### Salary Ranking According to Company Location  #####################

# IL (Illinois, USA): $271,446.50
# PR (Puerto Rico): $167,500.00
# US (USA Overall): $152,339.07
# RU (Russia): $140,333.33
# CA (Canada): $131,986.78
# NZ (New Zeland): $125,000.00
# BA (Bosnia Herzagovina): $120,000.00
# IE (İreland): $114,943.43
# JP (Japan): $114,127.33
# SE (Sweden): $105,000.00
average_salary = df.groupby('company_location')['salary'].mean().sort_values(ascending=False)

top_ten_countries = average_salary.head(10)

plt.figure(figsize=(15,10),dpi=80)

plt.bar(top_ten_countries.index, top_ten_countries)

plt.xlabel('Country')
plt.ylabel('Salary $')
plt.title('Average of the ten highest salaries by country')
plt.xticks(rotation=20, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()


############################ Categorizing salaries as a new feature  ############################

####Salaries have been categorized into low, medium, and high ##
def categorize_salary(salary):
    if salary < 80000:
        return 'Low'
    elif 80000 <= salary <= 180000:
        return 'Medium'
    else:
        return 'High'

df['salary_Level'] = df['salary'].apply(categorize_salary)
df.head(20)

df.groupby("company_size").agg({"salary":"mean"})
df.groupby("experience_level").agg({"salary": "mean"})
df.groupby("salary_Level").agg({"salary": ["mean","count"]})
df.groupby("employee_residence").agg({"salary": ["mean","count"]})


df["employee_residence"].value_counts()
df.head()

##################### Division by continents #####################

def get_country_name(alpha2_code):
    try:
        country = pycountry.countries.get(alpha_2=alpha2_code)
        return country.name
    except AttributeError:
        return None

df['Country_Fullname'] = df['employee_residence'].apply(get_country_name)

df['Continent'] = df['Country_Fullname'].apply(lambda x: cc.convert(x, to='continent'))

print(df)

df.head(100)

df["Country_Fullname"].isnull().sum()


############## Visualization by continents  ##################

df.groupby("job_title").agg({"Continent": "count"}).sort_values(by="Continent", ascending=False).head(10)

Continent_mean_salary= df.groupby('Continent')['salary'].mean().sort_values(ascending=False)
Continent_mean_salary.head(10)

plt.figure(figsize=(10, 6))
Continent_mean_salary.plot(kind='bar', color='skyblue')
plt.title('Kıtaya Göre Ortalama Maaş')
plt.xlabel('Kıta')
plt.ylabel('Ortalama Maaş')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


########### Salary trend change over the years ##########

plt.figure(figsize = (10,6))
salary_trend = df[['salary', 'work_year']].sort_values(by = 'work_year')
p = sns.lineplot(data =salary_trend ,x = 'work_year', y = 'salary', marker = 'o',linestyle='--', color='blue', markersize=8 )
plt.title('Salary Trend Over The Time', fontsize=12, fontweight='bold')

# Customize the background color
p.set_facecolor("#f4f4f4")
plt.legend(['Salary'], loc='best', fontsize=12)

# Remove the grid lines
p.grid(False)

plt.show()


######### Average salary by experience level ########

exp_salary = df.groupby('experience_level')['salary'].mean()
plt.figure(figsize = (10,6))
ax = sns.barplot(x = exp_salary.index, y = exp_salary.values, palette = 'Blues')
plt.title('Average Salary by Experience Level', fontsize=12, fontweight='bold')
plt.xlabel('Experience Level', fontsize=12, fontweight='bold')
plt.ylabel('Average Salary (USD)', fontsize=12, fontweight='bold')

for container in ax.containers:
    ax.bar_label(container,
                 padding=-50,
                 fontsize=17,
                 label_type="edge",
                 fontweight='bold'

                 )

ax.set_facecolor("#f4f4f4")
ax.grid(False)

plt.show()

############ Average salary by employment type ##############

emp_salary = df.groupby('employment_type')['salary'].mean()

plt.figure(figsize=(10, 6))
p = sns.barplot(y=emp_salary.values, x=emp_salary.index, palette='cool_r')
plt.title('Average Salary by Employment Type', fontsize=12, fontweight='bold')
plt.xlabel('Employment Type', fontsize=12, fontweight='bold')
plt.ylabel('Average Salary (USD)', fontsize=12, fontweight='bold')

for container in p.containers:
    plt.bar_label(container,
                  padding=-50,
                  fontsize=17,
                  label_type="edge",
                  fontweight='bold',
                  rotation=90

                  )

p.set_facecolor("#f4f4f4")
p.grid(False)
plt.show()


############################### Average salary and employment type by experience levels ##################################

cost_effectiveness = df.groupby(['experience_level', 'employment_type'])['salary'].mean().reset_index()

best_combination = cost_effectiveness.loc[cost_effectiveness['salary'].idxmax()]

plt.figure(figsize=(12, 8))

sns.set_palette(sns.color_palette('Blues'))

ax = sns.barplot(x='experience_level', y='salary', hue='employment_type', data=cost_effectiveness)
plt.xlabel('Experience Level', fontsize=14, fontweight='bold')
plt.ylabel('Average Salary (USD)', fontsize=14, fontweight='bold')
plt.title('Average Salary by Experience Level and Employment Type', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12, fontweight='bold')

for container in ax.containers:
    ax.bar_label(container, label_type="edge", color="black",
                 padding=6,
                 fontweight='bold',
                 fontsize=12,
                 bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': 'black'})


legend = plt.legend(title='Employment Type', fontsize=12, title_fontsize=14)
for text in legend.get_texts():
    text.set_fontweight('bold')

ax.set_facecolor("#f4f4f4")

ax.grid(False)

plt.show()



################################## The percentages of employees in the dataset by country #############################################################################

temp = df['Country_Fullname'].value_counts()[:10].to_frame()
temp.columns = ['count']
temp['percentage(%)'] =round(100 * df['Country_Fullname'].value_counts(normalize= True),2)

plt.figure(figsize=(10,6))
ax2 = sns.barplot(data = temp, y=temp.index,x=temp['percentage(%)'])
ax2.bar_label(ax2.containers[0])

plt.ylabel('Company Location')
plt.xlabel('Percentage(%)')
plt.suptitle('Company Location Count')
plt.show()
