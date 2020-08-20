#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Investigate a Dataset from TMDB Movies
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# 
# 
# > This Dataset has a record of around 10K movies, and has information on the budget and revenue, casts, director, and release date of the those movies.
# 
# #####  The dataset contains 21 columns. Below is the columns names and dicription:
# 
# | # | Column | Discription |
# | --- | --- | --- |
# | 01 | id | A id for each row |
# | 02 | imdb_id | Id given by the imbd system |
# | 03 | popularity | Popularity score |
# | 04 | budget | Budget recorded in dollars |
# | 05 | revenue | Revenue recorded in dollars |
# | 06 | original_title | The Movies' title |
# | 07 | cast | The cast that are in the movie |
# | 08 | homepage | A link to the hompage website |
# | 09 | director | The movie's director's name |
# | 10 | tagline | The movie's tagline |
# | 11 | keywords | Catchy short phrases for the movie|
# | 12 | overview | A short general overview of the story of the movie |
# | 13 | runtime | Movie's length in minutes |
# | 14 | genres | The movie's categories |
# | 15 | production_companies | The company which managed the process of the movie from the beginning till the end |
# | 16 | release_date | The release date of the movie |
# | 17 | vote_count | Total vote of the viewers |
# | 18 | vote_average | Average votes by the viewer |
# | 19 | release_year | The release year of the movie |
# | 20 | budget_adj | Budget of the associated movie in terms of 2010 dollars, accounting for inflation over time |
# | 21 | revenue_adj | Revenue of the associated movie in terms of 2010 dollars, accounting for inflation over time |
# 
# >### Questions to explore:
# > - What kinds of properties are associated with movies that have high revenues? in this question the focus will be on the **revenue_adj(dependant), and the 3 independants which are popularity, budget_adj, and vote_average** columns. 
# > - In which quarter of the year are most of the movies released, and does a particular quarter indicate a high revenue?
# 

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('darkgrid')
# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you document your steps carefully and justify your cleaning decisions.
# 
# ### General Properties

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('tmdb-movies.csv')
print (df.shape)
df.head(5)


# In[3]:


df.describe()


# In[37]:


df.hist(figsize=(16,10));


# > - From the graph above we can say that for Question 2 more movies were produced at the later years, but we will make sure when we analyze below.

# > - While trying to answer the first question I realised that a lot of budget and revenue and same goes for _adj, have a lot of "0's", which will ruin the results at the end so I will take the zero's and replace them with Nan so I can delete them with one line.

# In[4]:


df.info()


# > As we can see above, there are missing values in the **cast, homepage, director, tagline, keywords, overview, genres, production_companies** columns

# In[5]:


df.dtypes


# > As we can see above, there are two unique id's for each row, we can just keep the one called **"id"** as it is an int and will be easier to deal with
# 
# > Also we can see that the **"release_date"** column values are a object type, we will change them to date type in the data cleaning section

# In[6]:


df.nunique()


# > There is one not unique value in the id column, let see what it is

# In[7]:


not_unique = df['id'].duplicated()
not_unique.sum()


# >So, there is a duplicaed value in the **"id"** column, let us explore it

# In[8]:


df_not_unique = df[df['id'].duplicated()]
df_not_unique


# In[9]:


df[df['id'] == 42194]


# >So, the this is the duplicate, we are going to remove the duplicate in the data cleaning section

# In[10]:


df.duplicated().sum()


# > In the whole data we have only one duplicate, and we explored it in the previous cells

# 
# 
# ### Data Cleaning 
# 
# > First, let us drop the columns that are irrelevant to the proccess of exporing our questions and the second id for the rows **"imbd_id"**.
# > - For our first question (What kinds of properties are associated with movies that have high revenues?), we do not need the **homepage, tagline, keywords, and overview** columns
# > - Same goes for our second question (In which quarter of the year are most of the movies released?)
# 

# In[11]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
df.drop(['imdb_id', 'homepage', 'tagline', 'keywords', 'overview'], axis=1, inplace=True)


# In[12]:


df.head(1)


# > Now that droping is out of the way, let us convert **"release_date"** to a date data type

# In[13]:


df['release_date'] =  pd.to_datetime(df['release_date'])


# In[14]:


df.dtypes


# > Done, the data type of **"release_date"** has been changed to a date data type

# >Now let us see how we can deal with the remaining columns that have null or missing values

# In[15]:


df.info()


# > Now after dropping the uneeded columns, there are still the **cast, director, geners, & production_companies** columns, which are missing some values

# > For starters, our first question foucuses on the **popularity, budget_adj, and vote_average** columns.
# 
# > So for the following columns: **cast, director, geners, & production_companies** they have missing values, and we won't need them for our 2 questions, so lets store them and drop them from the original Dataset, so we can later merge them back, if we needed them. 
# 

# In[16]:


cast = df['cast']
director = df['director']
genres = df['genres']
prod_comp = df['production_companies']


# > Now that they are stored, let's drop them

# In[17]:


df.drop(['cast', 'director', 'genres', 'production_companies'], axis=1, inplace=True)


# In[18]:


print(df.shape)
df.head()


# In[19]:


df.info()


# > We have a budget and budget_adj columns, same goes for the revenue. the _adj counts for inflation, which would be more useful and more relevant to use. So let us drop the **budget, & revenue columns**

# In[20]:


df.drop(['budget', 'revenue'], axis=1, inplace=True)
df.head(1)


# In[21]:


df.nunique()


# > As we saw in the data wrangling section, there was 1 duplicated row, and we found it, so let's drop it now

# In[22]:


df.drop_duplicates(keep=False,inplace=True) 
print(df.shape)
print(df.duplicated().sum())


# > - As I mentioned before, I found later on that the budget_adj and revenue_adj have a lot of 0's which is not good for the analysis, so let's turn them into Nan and then delete them using: "df.dropna(axis=0, inplace=True)"

# In[23]:


df['budget_adj'].replace(0, np.NAN, inplace=True)
df['revenue_adj'].replace(0, np.NAN, inplace=True)


# In[24]:


df.describe()


# > - The issue is resolved.

# > The Data Cleaning is done.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1:
# > - What kinds of properties are associated with movies that have high revenues? in this question the focus will be on the **revenue_adj, popularity, budget_adj, vote_average** columns. 
# 

# In[25]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df.head()


# > - **revenue_adj** is the dependant variable, and the **popularity, budget_adj, and Vote_average** are the independant variables

# In[27]:


pd.plotting.scatter_matrix(df, figsize=(14,10));


# > - Initially we can see from the matrix above that there is some correlation between revenue_adj and the 3 variables we chose, but we need to further analyze below to make sure.

# > - Let's explore whether a low or a high vote average indicates high revenue

# In[28]:


median_vote = df['vote_average'].median()
# print(median_vote)

low_vote = df.query('vote_average < {}'.format(median_vote))
high_vote = df.query('vote_average >= {}'.format(median_vote))

mean_vote_low = low_vote['revenue_adj'].mean()
mean_vote_high = high_vote['revenue_adj'].mean()

print(mean_vote_low)
print(mean_vote_high)


# In[29]:


locations = [1, 2]
heights = [mean_vote_low, mean_vote_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('revenue by vote_average')
plt.xlabel('Vote')
plt.ylabel('revenue');


# > - We can see here that the higher vote indicates higher revenue

# > - let's now do the same with the popularity

# In[30]:


median_pop = df['popularity'].median()
print(median_pop)

low_pop = df.query('popularity < {}'.format(median_pop))
high_pop = df.query('popularity >= {}'.format(median_pop))

mean_pop_low = low_pop['revenue_adj'].mean()
mean_pop_high = high_pop['revenue_adj'].mean()

print(mean_pop_low)
print(mean_pop_high)


# In[31]:


locations = [1, 2]
heights = [mean_pop_low, mean_pop_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('revenue by popularity')
plt.xlabel('popularity')
plt.ylabel('revenue');


# > - We can uderstand from this that a higher vote_average indicates a higher revenue

# > - Let's now do the same with the budget_adj

# In[32]:


median_budget = df['budget_adj'].median()
print(median_budget)

low_budget = df.query('budget_adj < {}'.format(median_budget))
high_budget = df.query('budget_adj >= {}'.format(median_budget))

mean_budget_low = low_budget['revenue_adj'].mean()
mean_budget_high = high_budget['revenue_adj'].mean()

print(mean_budget_low)
print(mean_budget_high)


# In[33]:


locations = [1, 2]
heights = [mean_budget_low, mean_budget_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('revenue by budget_adj')
plt.xlabel('Budget')
plt.ylabel('revenue');


# > - As we can see, Higher budget is associated with higher revenue

# ### Research Question 2:
# > - Did the number of movies produced over the years increased, or decreased?

# > - Let's count number of movies in each year

# In[34]:


df.groupby('release_year').count()['original_title'].reset_index(name='counts').plot(x='release_year', y='counts', figsize=(14,10));


# > - The graph shows, that there was a significant increase of released movies since 1960

# In[35]:


df.groupby('release_year').count()['original_title'].reset_index(name='counts')


# > - We can see here that in 1960 there were 32 movies released, and in 2015 there were 629, shows that there was a significant increase in movies produced since 1960.

# # Conclusion

# In conclusion, in the first question we saw how **popularity, budget_adj, vote_average** were associated with the **revenue_adj**. It was found that a high movie popularity indicates a high revenue. Also the same conclusion was found for a high vote_average. And finally, we saw that a high budget_adj for a movie indicated a high revenue_adj.
# 
# In question 2 we determined whether more movies were produced over the years, or vice versa. And it was found that there was a significant increase in the movies produced compared to the early years.
# 
# But, there were a lot of limitations with the analyzing methods and the data set itself. There were a lot of missing data, and a lot of 0 values especially in the budget_adj, and revenue_adj columns, which had to be dropped so we can further analyze the data. Also there was no advanced statistics used which means that our findings are not to be taken as facts.

# In[38]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

