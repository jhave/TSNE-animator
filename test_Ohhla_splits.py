import matplotlib.pyplot as plt
import os, datetime
import re

import io

import itertools
from random import randint

#import BeautifulSoup

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer

######  PARAMETERS Randomized in loop of N iterations ##############

N=80000
init_ar = ["random","pca"]

# verbosity 
verb=0




#######   PATH   ############

READ_PATH = "Data/ohhla_CLEAN_txt_files_ALL/"



cnt =0
poems = []

ids=[]
filenames=[]
first_chars=[]

cnt=0
size_array=[]

nup=0
nup_orig=0


noty = 0


def solve_fast(s):
	ind1 = s.find('\n')
	#ind2 = s.rfind('\n')
	return s[ind1+4:]

print("Loading and parsing files:",READ_PATH)

for subdir, dirs, files in os.walk(READ_PATH):
	for file in files:
		if ".html" in file  and 'readme' not in file:
			if os.path.isfile(subdir+file):
				#print (subdir+file)

				nup_orig=nup_orig+1

				filenames.append(file.split(".txt.html")[0])
				#print("filenames: ",filenames[nup_orig-1])
				
				txt_data=open(subdir+file, encoding="latin-1").read()

				######## CLEAN 
				txt_data= txt_data.split("<pre>")[1]
				txt_data= txt_data.split("</pre>")[0]

				txt_data =txt_data.replace("&amp;","&")
				txt_data =txt_data.replace("Â","")

				######## START OF FINDING METADATA ###########

				txt_data =txt_data.replace("Title:","Song:")
				txt_data =txt_data.replace("Track:","Song:")
				txt_data =txt_data.replace("Song :","Song:")				
				txt_data =txt_data.replace("Song   ","Song:")
				txt_data =txt_data.replace("Song;","Song:")
				txt_data =txt_data.replace("Songs:","Song:")
				txt_data =txt_data.replace("SOng:","Song:")
				txt_data =txt_data.replace("Spng:","Song:")
				txt_data =txt_data.replace("song:","Song:")
				txt_data =txt_data.replace("Sont:","Song:")
				txt_data =txt_data.replace("Sonh:","Song:")

				artist = ''.join(txt_data.split("Album:")[0].lstrip("\n").split("Artist:")).strip("\n")
				album = ''.join(txt_data.split("Song:")[0].lstrip("\n").split("Album:")[-1]).strip("\n")
				song = ''.join(txt_data.split("Typed by:")[0].lstrip("\n").split("Song:")[-1]).strip("\n")

				if "\n" in song:
					song = song.split("\n")[0]
					#print("SHORTENING:",filenames[nup_orig-1],song)

				typed_by = ''.join(txt_data.split("Typed by:")[1].split("\n")[0])

				#print("Index: "+str(nup_orig-1),"\nFile: "+filenames[nup_orig-1]+".txt.html","\nArtist: "+artist,"\nAlbum:"+album,"\nSong: "+song,"\nTyped by: "+typed_by,"\n\n*****~!~*****\n")

				######## END OF FINDING METADATA ###########


				txt_data = ''.join(txt_data.split("Typed by:")[1]).split("\n",1)[1]
				#print(nup_orig-1,txt_data,"\n\n\n****************************\n\n\n")


				# with open("ohhla_CLEAN_LIST.txt", "a") as myfile:
				# 	myfile.write("Index: "+str(nup_orig-1)+"\nFile: "+filenames[nup_orig-1]+".txt.html"+"\nArtist: "+artist+"\nAlbum:"+album+"\nSong: "+song+"\nTyped by: "+typed_by+"\n\n*****~!~*****\n")


				# Artist and Album
				#AA = txt_data.split("Album:")[0]

				#artist = ''.join(AA.lstrip("\n").split("Artist:")).strip("\n")
				#album = ''.join(txt_data.split("Song:")[1].split("Album:")[-1]).strip("\n")

				# if len(album) > 200:
				# 	print("ALBUM*********************************\n\n\nfilenames: ",filenames[nup_orig-1],"      album: ",album[:40])
				#print("filenames: ",filenames[nup_orig-1],"      artist: ",artist)
				# if "<b>Warning</b>:" in artist:
				# 	os.remove(subdir+file)
				# 	print("filenames: ",filenames[nup_orig-1],"      artist: ",artist)

				# if len(artist)>260:
				# 	print("len(artist)",len(artist),"filenames: ",filenames[nup_orig-1],"      artist: ",artist) 
