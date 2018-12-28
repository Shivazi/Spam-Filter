
# coding: utf-8

# ### Importing Dependencies

# In[ ]:


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


# ### Image Extraction from URL

# In[ ]:


#site = 'https://www.tensorflow.org/tutorials/images/image_recognition'
print("Enter a URL..")
site=input()
response = requests.get(site)

soup = BeautifulSoup(response.text, 'html.parser')
img_tags = soup.find_all('img')

urls = [img['src'] for img in img_tags]
for url in urls:
    filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
    if filename!=None:
        with open(filename.group(1), 'wb') as f:
                if 'http' not in url:
                    # sometimes an image source can be relative 
                    # if it is provide the base url which also happens 
                    # to be the site variable atm. 
                    url = '{}{}'.format(site, url)
                response = requests.get(url)
                f.write(response.content)


# ### Png to Jpg Image

# In[ ]:


pngs = glob('./*.png')
for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite((j[:-3] + 'jpg' ), img)


# ### Model Load and Build

# In[ ]:


model = OpenNsfwModel()
model.build(weights_path= "data/open_nsfw-weights.npy", input_type=InputType.BASE64_JPEG)


# In[ ]:


fn_load_image = None
fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])


# In[ ]:


with tf.Session() as sess:
    jpgs = glob('./*.jpg')
    for j in range(len(jpgs)):
        if os.stat(jpgs[j]).st_size!=0:
            input_file =jpgs[j]
            sess.run(tf.global_variables_initializer())
            image = fn_load_image(input_file)
            predictions =sess.run(model.predictions,feed_dict={model.input: image})
            print("Results for '{}'".format(input_file))
            print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
            if((predictions[0][1])>.65):
                print("Nude Image")

            print("\n")
        
        
#Deleting the images
filelist = glob(os.path.join("*.png"))
filelist =filelist+ glob(os.path.join("*.jpg"))
for f in filelist:
    os.remove(f)
