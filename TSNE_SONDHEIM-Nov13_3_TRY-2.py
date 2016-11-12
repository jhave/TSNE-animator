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

READ_PATH = "Data/s_test/"  #"www.alansondheim.org/"



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
		if ".txt" in file  and 'readme' not in file:
			if os.path.isfile(subdir+file):
				#print (subdir+file)

				nup_orig=nup_orig+1

				filenames.append(file.split(".txt")[0])
				
				txt_data=open(subdir+file, encoding="latin-1").read()

				if file != "equations.txt" and  file != "equationselim.txt" and  file != "qa.txt"  and  file != "tg.txt"  and  file != "qr.txt":
					if file != "lp.txt" and file !="sn.txt"  and file !="re.txt"  and file !="pf.txt"  and file !="qs.txt":
						txt_data = txt_data.replace(r"===","----~~~~---")
						txt_data = txt_data.replace(r"====+","----~~~~---")
					txt_data_split = re.split(r"----~~~~---|-------------+|______________+|",txt_data)

				# Parse & store

				cnt=0
				for pt in txt_data_split:
					if len(pt.strip("\n"))>5:
						poems.append(pt.strip("\n"))
						cnt= cnt+1
						txt_fn = file.split(".txt")[0]+"_"+str(cnt)+".txt"
						ids.append(txt_fn)

						# LIMIT size of largest
						sz = len(pt.strip("\n"))/2000
						if sz < 4.0:
							sz = 4.0
						if sz > 300:
							sz = sz + (sz-300)/40

						sz = round(sz,2)

						size_array.append(sz)


						#SAVE SEGMENTS IN www.alansondheim.org_BDP_MIRROR
						# txt_fn = file.split(".txt")[0]+"_"+str(cnt)+".txt"

						# txt_fn_path = "Sondheim_BDP_MIRROR/"+txt_fn
						# f_txt=open(txt_fn_path,'w')
						# f_txt.write(pt)
						# f_txt.close() 


print ("# of Text segments:",len(poems))

# ANALYZE

vectors = TfidfVectorizer().fit_transform(poems)
print(repr(vectors))


# PARAMETERS INIT

lr_MAX = 600
lr_MIN = -600
lr_INIT = lr_MIN
lr_BOOL = True
lr_INC = 3

n_comp_init_MAX = 60
n_comp_init_MIN = 2
n_comp_init_BOOL = True
n_comp_init_INIT = n_comp_init_MIN
n_comp_init_INC = 2

complexity_gate =0

perp_MAX = 100
perp_MIN = 1
perp_INIT = perp_MIN
perp_BOOL = True
perp_INC = 1

exag_MAX = 11.0
exag_MIN = 1.0
exag_INIT = 1.0
exag_BOOL = True
exag_inc = 1

exag_gate =0

num_iter= 200
init = 'pca'

#layers? TODO: expand
n_comp = 2


# folder to SAVE into

save_PATH = str(datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M'))
if not os.path.exists("img/"+save_PATH):
	os.makedirs("img/"+save_PATH)
print("save_PATH:",save_PATH)

################### N iterations ######################

iters =0

for _ in itertools.repeat(None, N):

	iters = iters + 1
	print (str(iters))

	n_comp_init = n_comp_init_INIT
	lr = lr_INIT
	perp = perp_INIT
	exag = exag_INIT

	############# RAISE INIT PARAMETERS ##############

	if lr_INIT >lr_MAX:
		lr_BOOL = False
		lr_INIT =lr_MAX

		### PERPLEXITY
		if perp_BOOL:
			perp_INIT = perp_INIT + perp_INC #int(lr_INC/4)
		else:
			perp_INIT = perp_INIT - perp_INC #int(lr_INC/4)

			if perp_INIT >= perp_MAX or perp_INIT <= perp_MIN:
				perp_BOOL = not perp_BOOL

	elif lr_INIT < lr_MIN:
		lr_BOOL = True 
		lr_INIT =lr_MIN

	else:
		if lr_BOOL:
			lr_INIT = lr_INIT + lr_INC
		else:
			lr_INIT  = lr_INIT - lr_INC

		###

	complexity_gate = complexity_gate +1	
	if complexity_gate > 50 :

		complexity_gate =0

		if n_comp_init_BOOL:
			n_comp_init_INIT = n_comp_init_INIT + n_comp_init_INC #int(lr_INC/2)
		else:
			n_comp_init_INIT  = n_comp_init_INIT - n_comp_init_INC #int(lr_INC/2)


		if n_comp_init_INIT >= n_comp_init_MAX: 
			n_comp_init_BOOL = False
		elif n_comp_init_INIT <= n_comp_init_MIN: 
			n_comp_init_BOOL = True

		###

	exag_gate =exag_gate+1
	if exag_gate >11:
		exag_gate=0

		if exag_BOOL:
			exag_INIT = round( (exag_INIT + exag_inc),2)
		else:
			exag_INIT = round( (exag_INIT - exag_inc),2)

		if exag_INIT >= exag_MAX or exag_INIT <= exag_MIN:
			exag_BOOL = not exag_BOOL



	print ("************************************")
	print ("N_components (fed to TruncatedSVD) :",n_comp_init)
	print ("N_components (fed to TSNE) :",n_comp)
	print ("Perplexity (fed to TSNE) :",perp)
	print ("Learning_rate (fed to TSNE) :",lr)
	print ("Init(fed to TSNE) :",init)
	print ("Exaggeration:",exag)
	print ("Iterations:",num_iter)
	print ("************************************")

	# '''
	# For high-dimensional sparse data it is helpful to first reduce the dimensions to 50 dimensions with TruncatedSVD and then perform t-SN.head()E. This will usually improve the visualization.
	# '''

	X_reduced = TruncatedSVD(n_components=n_comp_init, random_state=0).fit_transform(vectors)
	X_embedded = TSNE(n_components=n_comp, perplexity=perp, learning_rate=lr, verbose=verb, init=init, early_exaggeration=exag, n_iter=num_iter).fit_transform(X_reduced)

	fig = plt.figure(figsize=(16, 9))
	fig.patch.set_facecolor('white')
	ax = plt.axes(frameon=False)
	plt.setp(ax, xticks=(), yticks=())
	plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
					wspace=0.0, hspace=0.0)

	nup = len(poems)

	plt.title('\n Sondheim files: %s \n Split into segments: %s  \n Learning rate: %s \n Complexity: %s  \n Perplexity: %s  \n Exaggeration: %s'%(nup_orig,nup,lr, n_comp_init,perp,exag), loc='left')
	
	# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=size_array[:],
	#         c='black', marker=".", alpha=1)
		
	plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=size_array[:],
			c='black', marker=".", alpha=1)

	fig.savefig("img/"+save_PATH+"/"+str(iters)+"_TSNE_SONDHEIM.png", transparent=False)
	print("IMAGE saved at: img/"+save_PATH+"/"+str(iters)+"_TSNE_SONDHEIM.png")

	for i, a in enumerate(filenames):
		ax.annotate(a, (X_embedded[:, 0][i], X_embedded[:, 1][i]))
		#ax.annotate(a+"\n'"+titles[i]+"'", (X_embedded[:, 0][i], X_embedded[:, 1][i]))

	#plt.show()
	fig.savefig("img/"+save_PATH+"/"+str(iters)+"_TSNE_SONDHEIM_ANNOTATED.png", transparent=False)


	X = X_embedded.tolist()
	for idx,x in enumerate(X):
		x.extend([ids[idx]])

	# SAVE to txt file as csv for later import to d3
	txt_fn = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')+"_SONDHEIM_TSNE_"+str(n_comp_init)+"_"+str(n_comp)+"_"+str(perp)+"_"+str(lr)+"_"+str(exag)+"_num_iter"+str(num_iter)+".txt"

	txt_fn_path = "txt/"+txt_fn
	f_txt=open(txt_fn_path,'w')
	f_txt.write("x,y,id,s")

	inc=0
	for x in X:
		f_txt.write("\n")

		for idx,item in enumerate(x):
			if idx == 2:
				f_txt.write("\"%s\"," % item)
			else:
				f_txt.write("%.2f," % item)

		f_txt.write("%s" % size_array[inc])
		inc=inc+1

	f_txt.close();   

	print("\nTXT file created at:",txt_fn_path,"\n")



# Parameters
#  |  ----------
#  |  n_components : int, optional (default: 2)
#  |      Dimension of the embedded space.
#  |  
#  |  perplexity : float, optional (default: 30)
#  |      The perplexity is related to the number of nearest neighbors that
#  |      is used in other manifold learning algorithms. Larger datasets
#  |      usually require a larger perplexity. Consider selcting a value
#  |      between 5 and 50. The choice is not extremely critical since t-SNE
#  |      is quite insensitive to this parameter.
#  |  
#  |  early_exaggeration : float, optional (default: 4.0)
#  |      Controls how tight natural clusters in the original space are in
#  |      the embedded space and how much space will be between them. For
#  |      larger values, the space between natural clusters will be larger
#  |      in the embedded space. Again, the choice of this parameter is not
#  |      very critical. If the cost function lr_INCeases during initial
#  |      optimization, the early exaggeration factor or the learning rate
#  |      might be too high.
#  |  
#  |  learning_rate : float, optional (default: 1000)
#  |      The learning rate can be a critical parameter. It should be
#  |      between 100 and 1000. If the cost function lr_INCeases during initial
#  |      optimization, the early exaggeration factor or the learning rate
#  |      might be too high. If the cost function gets stuck in a bad local
#  |      minimum lr_INCeasing the learning rate helps sometimes.
#  |  
#  |  n_iter : int, optional (default: 1000)
#  |      Maximum number of iterations for the optimization. Should be at
#  |      least 200.
#  |  
#  |  metric : string or callable, (default: "euclidean")
#  |      The metric to use when calculating distance between instances in a
#  |      feature array. If metric is a string, it must be one of the options
#  |      allowed by scipy.spatial.distance.pdist for its metric parameter, or
#  |      a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
#  |      If metric is "precomputed", X is assumed to be a distance matrix.
#  |      Alternatively, if metric is a callable function, it is called on each
#  |      pair of instances (rows) and the resulting value recorded. The callable
#  |      should take two arrays from X as input and return a value indicating
#  |      the distance between them.
#  |  
#  |  init : string, optional (default: "random")
#  |      Initialization of embedding. Possible options are 'random' and 'pca'.
#  |      PCA initialization cannot be used with precomputed distances and is
#  |      usually more globally stable than random initialization.
#  |  
#  |  verbose : int, optional (default: 0)
#  |      Verbosity level.
#  |  
#  |  random_state : int or RandomState instance or None (default)
#  |      Pseudo Random Number generator seed control. If None, use the
#  |      numpy.random singleton. Note that different initializations
#  |      might result in different local minima of the cost functio

