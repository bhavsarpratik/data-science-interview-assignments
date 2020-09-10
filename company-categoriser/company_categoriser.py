from bs4 import BeautifulSoup
import re
import wikipedia
import nltk
from nltk.corpus import stopwords
import urllib3
import pandas as pd
urllib3.disable_warnings()

categoriesURL = "https://www.g2crowd.com/categories?category_type=software"


def getSoup(url):  # function to get HTML

    http = urllib3.PoolManager()
    response = http.request('GET', url)
    return BeautifulSoup(response.data, 'html.parser')

# NLTK corpus of stopwords and english words
stopWords = set(stopwords.words('english'))
englishWords = nltk.corpus.words.words()


def getSoupForAnyWebsite(url):
    '''Removes all the junk from HTML. Sentences are lost because of set function'''

    http = urllib3.PoolManager()
    response = http.request('GET', url)
    words = set(BeautifulSoup(response.data,
                              'html.parser').get_text().lower().split())

    words = set(w for w in words if not w in stopWords)
    words = set(w for w in words if w in englishWords)
    return ' '.join(words)

title, url, categoryData = [], [], []


def getCategories():
    '''Gets all li from the g2crowd page'''

    soup = getSoup(categoriesURL)

    for ul in soup.find_all('ul', {'class': 'categories-list lvl-1'}):
        for li in ul.find_all('li'):
            link = li.find('a').get_text()
            if link not in title:
                title.append(link)
                url.append('https://www.g2crowd.com' + link['href'])

getCategories()

# filtering unwanted URLs and scraping the necessary text from description page. URLs
# are removed based on presence of specific class(description class of webpage)
categoryInfo, workingURLs, workingTitle = [], [], []

for i, link in enumerate(url):
    text = []
    soup = getSoup(link)
    try:
        data = soup.find("div", {'class': [
                         'row columns no-padding-x hide-for-small-only', 'medium-6 columns hide-for-small-only']})
        text = data.get_text().replace('\n', ' ')
        length = len(text)
        categoryInfo.append(text)
        workingURLs.append(link)
        workingTitle.append(title[i])
    except:
        print('Unwanted link:', link)


# Some data is blank in categoryInfo. Hence getting data for it by running
# loop over categoryInfo
print('Length of categoryInfo is', len(categoryInfo))
for i, ci in enumerate(categoryInfo):
    if ci == '':
        print(i, 'is blank')
        try:
            soup = getSoup(allCategoryURL[i])
            data = soup.find("div", {'class': [
                             'row columns no-padding-x hide-for-small-only', 'medium-6 columns hide-for-small-only']})
            categoryInfo[i] = data.get_text().replace('\n', ' ')
            print(i, 'is updated')
        except:
            print('err at', i, allCategoryURL[i])

    if categoryInfo[i] == '':
        categoryInfo.pop(i)
        allCategoryURL.pop(i)
        title.pop(i)
        print(i, 'is removed')

print('New length of categoryInfo is', len(categoryInfo))


# Printing categories for all the companies in our text file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities, related_docs_indices = [], []


def getCategory(data):
    '''Function gives category of a company using cosine value'''

    tfidf = TfidfVectorizer().fit_transform(data)
    vect = TfidfVectorizer(min_df=1, stop_words='english')
    tfidf = vect.fit_transform(data)
    cosineSimilarities = linear_kernel(tfidf[0:1], tfidf).flatten()
    # finding top 5 similar articles
    relatedDocsIndices = cosineSimilarities.argsort()[:-5:-1]
    # print(related_docs_indices)
    # print(cosine_similarities[related_docs_indices])
    return relatedDocsIndices


def categoryFromWebsite(name):
    '''sub-function of project that extracts company info from website and
    gives category'''

    info = ' '
    url = 'www.' + name + '.com'
    try:
        info = getSoupForAnyWebsite(url)
        # adding new document to array at 0th location for TF-IDF
        categoryInfo.insert(0, info)
        category = getCategory(categoryInfo)
        # removing new document from array from 0th location
        categoryInfo.pop(0)
        return category
    except:
        print('Website not available:', url)


def categoryFromWikipedia(name, info):
    '''sub-function of project that extracts company info from wikipedia and
    gives category'''

    info = wikipedia.summary(name).lower()
    if re.search(r'%s' % name, info):
        # adding new document to array at 0th location for TF-IDF
        categoryInfo.insert(0, info)
        category = getCategory(categoryInfo)
        # removing new document from array from 0th location
        categoryInfo.pop(0)
        return category
        else:
            return categoryFromWebsite(name)


def project(name):
    '''final function for getting company category'''

    name = str(name)

    info = ' '
    name = name.lower().strip()
    try:
        # using wikipedia api to get company info. If wikipedia fails, we will
        # try to get info from the company website

        category = categoryFromWikipedia(name)

    except:
        print('No Wiki info available on', name)
        category = categoryFromWebsite(name)

    if category:
        print('Company - %s \n Category: \n 1. %s - %s \n OR \n 2.%s - %s' % (name, title[category[
              1] - 1], allCategoryURL[category[1] - 1], title[category[2] - 1], allCategoryURL[category[2] - 1]))

# reading the txt file containing list of companies
with open(os.path.join(os.path.dirname(__file__), 'company_list.txt')) as f:
    companies = f.readlines()

# getting category for all companies
for company in companies:
    project(company)
