
# coding: utf-8

# ### Spam Inference Engine

# In[49]:


#importing dependencis
import numpy as np
import os


# ### Feature Extractions

# In[50]:


def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in sorted(os.listdir(mail_dir))]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
        with open(fil) as fi:
            for i,line in enumerate(fi):
                if i == 2:
                    words = line.split() 
                    for word in words:
                        wordID = 0 
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix


# ### Loading Model

# In[51]:


import pickle
with open("model1_SVC",'rb') as file1:
    model1_SVC=pickle.load(file1)
with open("model2_LNB",'rb') as file2:
    model2_LNB=pickle.load(file2)
#Loading dictionary data
with open('dictionary','rb') as file:
    dictionary= pickle.load(file)


# ### Text Extraction 

# In[52]:


import urllib.request
import inscriptis
# url="https://docs.python.org/3.8/about.html"
print("Enter a URL..")
url=input()
html = urllib.request.urlopen(url).read().decode('utf-8')
text = inscriptis.get_text(html)

words=text.split()
list_build=[]
for i in range(len(words)):
    if len(words[i])<=3:
        list_build.append(words[i]) 
for i in range(len(list_build)):
    words.remove(list_build[i])

#list to string generation
words=" ".join(words)

#write text file in a folder
import random
textfile = open('/home/nybsys/Desktop/Spam Filter/ling-spam/Predicted_mail/textfile.txt', 'w')
textfile.write(words)
textfile.close()


# ### Spam Prediction

# In[53]:


# Test the unseen mails for Spam
test_dir = '/home/nybsys/Desktop/Spam Filter/ling-spam/Predicted_mail'
test_matrix = extract_features(test_dir)
#result1 = model1_SVC.predict(test_matrix)
result2 = model2_LNB.predict(test_matrix)
if(result2==0):
    print("Not spam text")
else: print("Spam text")

