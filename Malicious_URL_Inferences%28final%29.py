
# coding: utf-8

# ## Loading pickle file

# In[1]:
import pickle
# Load from file
pkl_filename="pickle_model_URL.pkl"
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)


# In[2]:


import pickle
# Load from file
pkl_filename="pickle_model_vectorizer.pkl"
with open(pkl_filename, 'rb') as file:  
    pickle_model_vectorizer = pickle.load(file)


# ## HTML Parsing

# In[3]:


import requests
#from BeautifulSoup import BeautifulSoup
import bs4


# ## URL Extraction

# In[4]:


def getURL(page):
    
    """
    :param page: html of web page (here: Python home page) 
    :return: urls in that page 
    """
    
    start_link = page.find("a href")
    if start_link == -1:
        return None, 0
    start_quote = page.find('"', start_link)
    #print(start_quote)
    end_quote = page.find('"', start_quote + 1)
    #print(end_quote)
    url = page[start_quote + 1: end_quote]
    return url, end_quote


# ## Checking Spam

# In[9]:


while True:
    print("Enter a Url...")
    url = input()
    if url=='close':
        break
    response = requests.get(url)
    #print(response)
    # parse html
    page = str(bs4.BeautifulSoup(response.content,"lxml"))

    bad_cnt=0
    good_cnt=0
    while True:
        url, n = getURL(page)
        page = page[n:]
        if url:
            if url.find("https")==0:
                url=url[8:]
                url=[url]
                #print(url)
                X_predict = pickle_model_vectorizer.transform(url)
                New_predict = pickle_model.predict(X_predict)
                #print(New_predict)
                if New_predict=="bad":
                    bad_cnt=bad_cnt+1
                else: good_cnt=good_cnt+1
            elif url.find("http")==0 and url.find("https")!=0:
                url=url[7:]
                url=[url]
                #print(url)
                #vectorizer.fit_transform(url)
                X_predict = pickle_model_vectorizer.transform(url)
                New_predict = pickle_model.predict(X_predict)
                #print(New_predict)
                if New_predict=="bad":
                    bad_cnt=bad_cnt+1
                else: good_cnt=good_cnt+1

        else:
            break


    if good_cnt>=bad_cnt:
        print("Not Spam")
    else :
        print("Spam")
    print(bad_cnt, good_cnt)
else:
    system.exit()

