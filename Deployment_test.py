
# coding: utf-8

# ## NFSW Detection

# In[1]:


import re
import os
import cv2
import requests
from glob import glob
from bs4 import BeautifulSoup

import sys
import base64
import tensorflow as tf
import numpy as np
from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader


# ### Sign In mail

# In[2]:


import time
import email
import smtplib
import imaplib
import getpass

ORG_EMAIL   = "@gmail.com"
print("Enter email password...")
email="shivazibiswas.ice.iu"
FROM_EMAIL  = email + ORG_EMAIL
#FROM_PWD = getpass.getpass("PASSWORD : ")
FROM_PWD="studyhardstaycool"
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT   = 993 


# ### Mail Body and Image Extraction

# In[3]:


import email
mail = imaplib.IMAP4_SSL(SMTP_SERVER)
mail.login(FROM_EMAIL,FROM_PWD)
mail.select('inbox')

type, data = mail.search(None, 'ALL')
mail_ids = data[0]
id_list = mail_ids.split()   

latest_email_id = int(id_list[-1])
typ,data=mail.fetch(str.encode(str(latest_email_id)), '(RFC822)' )

for response_part in data:
    if isinstance(response_part, tuple):
        #getting mime message
        msg=email.message_from_string(response_part[1].decode('utf-8'))
        messageMainType = msg.get_content_maintype()
        
        
        html_data=""
        plain_data=""
        if messageMainType== 'multipart':
            for part in msg.get_payload():
                if part.get_content_subtype()=='html':
                    html_data=html_data+part.get_payload()
                elif part.get_content_subtype()=='plain':
                    plain_data=plain_data+part.get_payload()
        elif messageMainType=='text':
            html_data=msg.get_payload()
            plain_data=msg.get_payload()
            

email_subject = msg['subject']
email_from = msg['from']
email_to=msg['to']

print("\n")
print ('From : ' + email_from + '\n')
print ('To : ' + email_to + '\n')
print ('Subject : ' + email_subject + '\n')


            
##Image extraction            
soup = BeautifulSoup(html_data,"lxml")
#print(plain_data)
#print(soup)

import quopri
mystring = html_data
mystring=mystring.encode('utf-8')
decoded_string = quopri.decodestring(mystring)
decoded_string=BeautifulSoup(decoded_string)
#print(decoded_string)

img_tags = decoded_string.find_all('img')
#img_tags=soup.find_all('img')
#print(img_tags)
urls = [img['src'] for img in img_tags]
#print(urls)

for url in urls:
    filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
    if filename!=None:
        with open(filename.group(1), 'wb') as f:
            response = requests.get(url)
            f.write(response.content)


# ### Png to Jpg Image

# In[4]:


pngs = glob('./*.png')
for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite((j[:-3] + 'jpg' ), img)


# ### Model Load and Build

# In[5]:


model = OpenNsfwModel()
model.build(weights_path= "data/open_nsfw-weights.npy", input_type=InputType.BASE64_JPEG)


# In[6]:


fn_load_image = None
fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])


# ### Session And Run

# In[7]:


#All full black image delete
# jpgs = glob('./*.jpg')
# # print(jpgs)
# for j in range(len(jpgs)):
#     #print(jpgs[j])
#     #print(os.path.getsize(jpgs[j]))
#     #print("yes")
#     image = cv2.imread(jpgs[j], 0)
#     if cv2.countNonZero(image) == 0:
#         os.remove(jpgs[j])


# In[8]:


Nude_Image=0
Non_Nude_Image=0
with tf.Session() as sess:
    jpgs = glob('./*.jpg')
    for j in range(len(jpgs)):
        if os.stat(jpgs[j]).st_size !=0:
            input_file =jpgs[j]
            sess.run(tf.global_variables_initializer())
            image = fn_load_image(input_file)
            predictions =sess.run(model.predictions,feed_dict={model.input: image})
            #print("Results for '{}'".format(input_file))
            #print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
            if((predictions[0][1])>.65):
                Nude_Image=Nude_Image+1
            else: Non_Nude_Image=Non_Nude_Image+1

Extracted_Image=Nude_Image+Non_Nude_Image
# print("\n")
# print("Nude_Image: '{}'" .format(Nude_Image))
# print("Non_Nude_Image: '{}'" .format(Non_Nude_Image))

# if Nude_Image==0:
#     print("Extracted Images from URL has been predicted as Hum")
# else: print("Extracted Images from URL has been predicted as Spam")


# ### Remove Images

# In[9]:


filelist = glob(os.path.join("*.png"))
filelist =filelist+ glob(os.path.join("*.jpg"))
for f in filelist:
    os.remove(f)


# ## Malicious URL Detection

# In[10]:


import pickle
# Load from file
pkl_filename="pickle_model_URL.pkl"
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)


# In[11]:


import pickle
# Load from file
pkl_filename="pickle_model_vectorizer.pkl"
with open(pkl_filename, 'rb') as file:  
    pickle_model_vectorizer = pickle.load(file)


# In[12]:


import requests
import bs4


# ### URL Extraction

# In[13]:


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


# ### Checking Spam

# In[14]:


#url = site
#response = requests.get(url)
#print(response)
# parse html

#print(decoded_string)
#page = str(bs4.BeautifulSoup(decoded_string,"lxml"))
page=str(decoded_string)
bad_cnt=0
good_cnt=0
while True:
    url, n = getURL(page)
    #print(url)
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

Extracted_Url=bad_cnt+good_cnt

# print("\n")
# print("Prediction_of_Reliable_URL: '{}' " .format(good_cnt))        
# print("Prediciton_of_Malicious_URL: '{}' " .format(bad_cnt))

# if bad_cnt>0:
#     print("Extracted URL has been predicted as Spam")
# else: print("Extracted URL has been predicted as Hum")


# ## Text-Spam Inference Engine

# In[15]:


#importing dependencis
import numpy as np
import os


# ### Feature Extraction

# In[16]:


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

# In[17]:


import pickle
with open("model1_SVC",'rb') as file1:
    model1_SVC=pickle.load(file1)
with open("model2_LNB",'rb') as file2:
    model2_LNB=pickle.load(file2)
#Loading dictionary data
with open('dictionary','rb') as file:
    dictionary= pickle.load(file)


# ### Text Extraction

# In[18]:


# import urllib.request
# import inscriptis
# url="https://docs.python.org/3.8/about.html"
#print("Enter a URL..")

# url=site
# html = urllib.request.urlopen(url).read().decode('utf-8')
# text = inscriptis.get_text(html)

import random

if Extracted_Image==0 and Extracted_Url==0:
    #write text file in a folder
    textfile = open('/home/nybsys/Desktop/Deployment/Predicted_mail/textfile.txt', 'w')
    textfile.write(plain_data)
    textfile.close()

else:

    text=soup.text
    #print(text)

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
    textfile = open('/home/nybsys/Desktop/Deployment/Predicted_mail/textfile.txt', 'w')
    textfile.write(words)
    textfile.close()


# ### Spam Prediction

# In[19]:


# Test the unseen mails for Spam
test_dir = '/home/nybsys/Desktop/Deployment/Predicted_mail'
test_matrix = extract_features(test_dir)
#result1 = model1_SVC.predict(test_matrix)
result2 = model2_LNB.predict(test_matrix)

# if(result2==0):
#     print("Extracted text has been predicted as Hum")
# else: print("Extracted text has been predicted as Spam")


# # Final Comment

# In[20]:


#print("\nTargeted URL...")
#print(site)

# print("\n")
# print("Prediction_of_Reliable_URL: '{}' " .format(good_cnt))        
# print("Prediciton_of_Malicious_URL: '{}' " .format(bad_cnt))

# print("\n")    
# print("Prediction of Non_Nude_Image: '{}'" .format(Non_Nude_Image))
# print("Prediction of Nude_Image: '{}'" .format(Nude_Image))

if Nude_Image==0:
    print("\nExtracted Image from URL has been predicted as Hum")
else: print("\nExtracted Image from URL has been predicted as Spam")
    
if bad_cnt>good_cnt:
    print("Extracted URL has been predicted as Spam")
else: print("Extracted URL has been predicted as Hum")
    
if(result2==0):
    print("Extracted text from URL has been predicted as Hum")
else: print("Extracted text from URL has been predicted as Spam")
    
print("\n")
if Nude_Image==0 and bad_cnt<=good_cnt and result2==0:
    print("Final_Comment: Hum")
else: print("Final_Comment: Spam")
    
    
if Nude_Image==0 and bad_cnt<=good_cnt and result2==0:
    URL="http://120.50.14.28:8080/mail/send"
    requests.post(URL,json={"from": email_from,"to": email_to,"subject": email_subject,"msg": plain_data})

