import matplotlib.pyplot as plt
import os, datetime
import re

import io

import itertools
from random import randint

#import BeautifulSoup
from bs4 import BeautifulSoup, NavigableString, Tag

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer

######  PARAMETERS Randomized in loop of N iterations ##############

N=80000
init_ar = ["random","pca"]

# verbosity 
verb=0




#######   PATH   ############

READ_PATH = "Data/Sophie_leaving_EVERNOTE_EXPORT_HTML/"#"www.alansondheim.org/"



cnt =0
poems = []

ids=[]
filenames=[]
first_chars=[]

cnt=0
size_array=[]

nup=0
nup_orig=0

print("Loading and parsing files:",READ_PATH)

for subdir, dirs, files in os.walk(READ_PATH):
	for file in files:
		if ".html" in file  and 'readme' not in file:
			if os.path.isfile(subdir+file):
				print (subdir+file)

				nup_orig=nup_orig+1
				
				filenames.append(file.split(".html")[0])
			
				txt_data=open(subdir+file, encoding="latin-1").read()
				txt_data = txt_data.replace("Â","")
				

				soup = BeautifulSoup(txt_data, 'html.parser')
				
				txt_data = soup.title.string+"\n"+soup.get_text()
				#print(txt_data)

				for br in soup.findAll('br'):
					next = br.nextSibling
					if not (next and isinstance(next,NavigableString)):
						continue
					next2 = next.nextSibling
					if next2 and isinstance(next2,Tag) and next2.name == 'br':
						text = str(next).strip()
						if text:
							print ("Found:", next)
							poems.append(next)
							cnt= cnt+1
							txt_fn = file.split(".txt")[0]+"_"+str(cnt)+".txt"
							ids.append(txt_fn)

							# LIMIT size of largest
							sz = len(next)
							if sz < 1.0:
								sz = 1.0
							if sz > 300:
								sz = sz + (sz-300)/40

							sz = round(sz,2)


