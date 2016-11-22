import matplotlib.pyplot as plt
import os, datetime
import re

import io
import argparse

import itertools
from random import randint

#import BeautifulSoup

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer

######  PARAMETERS Randomized in loop of N iterations ##############

N=8000
init_ar = ["random","pca"]

# verbosity 
verb=0




#######   PATH   ############

READ_PATH = "Sondheim_BDP_MIRROR/"



cnt =0
poems = []

ids=[]
first_chars=[]

cnt=0
size_array=[]

nup=0
nup_orig=0

cnt=0


print("Loading and parsing files:",READ_PATH)

for subdir, dirs, files in os.walk(READ_PATH):
	for file in files:
		if ".txt" in file:
			if os.path.isfile(subdir+file):
				print (cnt,subdir+file)

				nup_orig=nup_orig+1
				
				txt_data=open(subdir+file).read()
				poems.append(txt_data)
				
				# Parse & store
				cnt= cnt+1
				txt_fn = file.split(".txt")[0]
				ids.append(txt_fn)

				# LIMIT size of largest
				sz = len(txt_data.strip("\n"))/2000
				if sz < 4.0:
					sz = 4.0
				if sz > 300:
					sz = sz + (sz-300)/40

				sz = round(sz,2)

				size_array.append(sz)


print ("# of Text segments:",len(poems))

# ANALYZE

vectors = TfidfVectorizer().fit_transform(poems)
print(repr(vectors))



# PARAMETERS INIT

lr_MAX = 1100
lr_MIN = -600
#lr_INIT = lr_MIN
lr_BOOL = True
lr_INC = 0.1

lr_gate = 0

n_comp_init_MAX = 160
n_comp_init_MIN = 2
n_comp_init_BOOL = True
complexity_INIT = n_comp_init_MIN
complexity_INC = 0.1

complexity_gate = 0

perp_MAX = 120
perp_MIN = 1
#perp_INIT = perp_MIN
perp_BOOL = True
perp_INC = 1.0

perplexity_gate = 0

#exag_MAX = 33.0
exag_MIN = 1.0
#exag_INIT = 1.0
exag_BOOL = True
exag_inc = 0.1

exag_gate =0


# How often will parameters change: 'limit = 0' means change always
lr_gate_limit =0


#
c_gate_limit = 0
p_gate_limit = 0


num_iter= 200
init = 'pca'

## LAYERS
n_comp = 2



################### Get Args ##########################

parser = argparse.ArgumentParser()


parser.add_argument('--lr_gate_limit', type=int, default=0)
parser.add_argument('--e_gate_limit', type=int, default=10)
parser.add_argument('--c_gate_limit', type=int, default=20)

parser.add_argument('--learning_rate', type=float, default=-600.0)
parser.add_argument('--learning_rate_INC', type=float, default=0.1)

parser.add_argument('--exag_rate', type=float, default=1.0)
parser.add_argument('--exag_rate_INC', type=float, default=0.1)
parser.add_argument('--exag_MAX', type=float, default=33)

parser.add_argument('--complexity', type=int, default=50)
parser.add_argument('--complexity_INC', type=float, default=0.1)

parser.add_argument('--perplexity', type=float, default=20)
parser.add_argument('--perplexity_INC', type=float, default=0.1)
parser.add_argument('--perplexity_MIN', type=float, default=1)
parser.add_argument('--perplexity_MAX', type=float, default=80)

parser.add_argument('--random_state', type=int, default=None)

args = parser.parse_args()

#print("--learning_rate:",args.learning_rate)

lr_INIT = args.learning_rate
lr_INC = args.learning_rate_INC

exag_INIT = args.exag_rate
exag_inc = args.exag_rate_INC
exag_MAX = args.exag_MAX

e_gate_limit = args.e_gate_limit
c_gate_limit = args.c_gate_limit
lr_gate_limit = args.lr_gate_limit

complexity_INIT = args.complexity
complexity_INC = args.complexity_INC

perp_INIT = args.perplexity
perp_INC = args.perplexity_INC
perp_MIN = args.perplexity_MIN
perp_MAX = args.perplexity_MAX

random_state = args.random_state


##################  SAVE Folders ##############

dt =datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')
save_PATH = "SONDHEIM_LR{}_EXAG{}_C{}_EGL{}_{}".format(lr_INIT, exag_INIT, complexity_INIT, e_gate_limit, dt)
if not os.path.exists("img/"+save_PATH):
	os.makedirs("img/"+save_PATH)
print("save_PATH:",save_PATH)

t_save_PATH = "SONDHEIM_{}".format( dt)
txt_save_PATH = "txt/{}/".format(t_save_PATH)
if not os.path.exists(txt_save_PATH):
	os.makedirs(txt_save_PATH)





################### N iterations ######################

iters =0


for _ in itertools.repeat(None, N):

	iters = iters + 1
	print (str(iters))


	lr = lr_INIT
	perp = perp_INIT
	exag = exag_INIT



	############# RAISE INIT PARAMETERS ##############


	### LEARNING RATE
	lr_gate = lr_gate +1	
	if lr_gate > lr_gate_limit :

		lr_gate =0
		if lr_INIT >lr_MAX:
			lr_BOOL = False
			lr_INIT =lr_MAX

		elif lr_INIT < lr_MIN:
			lr_BOOL = True 
			lr_INIT =lr_MIN

		else:
			if lr_BOOL:
				lr_INIT = round(lr_INIT + lr_INC,2)
			else:
				lr_INIT  = round(lr_INIT - lr_INC,2)

	# ### PERPLEXITY
	perplexity_gate = perplexity_gate +1	
	if perplexity_gate > p_gate_limit :

		perplexity_gate =0

		if perp_BOOL:
			perp_INIT = round( (perp_INIT + perp_INC),2)
		else:
			perp_INIT = round( (perp_INIT - perp_INC),2)

		if perp_INIT >= perp_MAX: 
			perp_BOOL = False
		elif perp_INIT <= perp_MIN: 
			perp_BOOL = True

	### COMPLEXITY (reduction at beginning)

	complexity_gate = complexity_gate +1
	#print(complexity_gate,c_gate_limit, complexity_INIT, complexity_INIT)	
	if complexity_gate > c_gate_limit :
		#print("inside complex",n_comp_init_BOOL,complexity_INC)
		complexity_gate =0

		if n_comp_init_BOOL:
			complexity_INIT= round( (complexity_INIT + complexity_INC),2)
			#print("+ complexity_INIT",complexity_INIT,complexity_INC)
		else:
			complexity_INIT  = round( (complexity_INIT - complexity_INC),2)
			#print("- complexity_INIT",complexity_INIT,complexity_INCplex)

		

		if complexity_INIT >= n_comp_init_MAX: 
			n_comp_init_BOOL = False
		elif complexity_INIT <= n_comp_init_MIN: 
			n_comp_init_BOOL = True

	### Early Exaggeration

	exag_gate =exag_gate+1
	if exag_gate >e_gate_limit:
		exag_gate=0

		if exag_BOOL:
			exag_INIT = round( (exag_INIT + exag_inc),2)
		else:
			exag_INIT = round( (exag_INIT - exag_inc),2)

		if exag_INIT >= exag_MAX or exag_INIT <= exag_MIN:
			exag_BOOL = not exag_BOOL



	print ("************************************")
	print ("N_components (fed to TruncatedSVD) :",complexity_INIT)
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

	X_reduced = TruncatedSVD(n_components=complexity_INIT, random_state=0).fit_transform(vectors)
	X_embedded = TSNE(n_components=n_comp, perplexity=perp, learning_rate=lr, verbose=verb, init=init, early_exaggeration=exag, n_iter=num_iter, random_state=random_state).fit_transform(X_reduced)

	fig = plt.figure(figsize=(16, 9))
	fig.patch.set_facecolor('white')
	ax = plt.axes(frameon=False)
	plt.setp(ax, xticks=(), yticks=())
	plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
					wspace=0.0, hspace=0.0)

	nup = len(poems)

	plt.title('LYS: %s | LR: %s \nC: %s | P: %s | E: %s'%(n_comp,lr, complexity_INIT,perp,exag), loc='left')
			
	plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=size_array[:],
			c='black', marker=".", alpha=1)

	fig.savefig("img/"+save_PATH+"/TSNE_SONDHEIM_{0:05d}.png".format(iters), transparent=True, figsize=(16.0, 9.0), dpi=320)
	print("IMAGE saved at: img/"+save_PATH+"/TSNE_SONDHEIM_{0:05d}.png".format(iters))

	# for i, a in enumerate(txt_fn):
	# 	ax.annotate(a, (X_embedded[:, 0][i], X_embedded[:, 1][i]))
	# 	#ax.annotate(a+"\n'"+titles[i]+"'", (X_embedded[:, 0][i], X_embedded[:, 1][i]))

	# #plt.show()
	# fig.savefig("img/"+save_PATH+"/"+str(iters)+"_TSNE_SONDHEIM_ANNOTATED.png", transparent=False, figsize=(16.0, 9.0), dpi=320)


	X = X_embedded.tolist()
	for idx,x in enumerate(X):
		x.extend([ids[idx]])

	# SAVE to txt file as csv for later import to d3
	#txt_fn = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')+"_SONDHEIM_TSNE_"+str(complexity_INIT)+"_"+str(n_comp)+"_"+str(perp)+"_"+str(lr)+"_"+str(exag)+"_num_iter"+str(num_iter)+".txt"

	txt_fn = "LR{}_EXAG{}_C{}_LAYERS{}_P{}.txt".format(lr,exag,complexity_INIT,n_comp,perp)


	txt_fn_path = txt_save_PATH+txt_fn
	f_txt=open(txt_fn_path,'w')
	f_txt.write("x,y,id,s")

	inc=0
	for x in X:
		f_txt.write("\n")

		if n_comp == 2:
			for idx,item in enumerate(x):
				if idx == 2:
					f_txt.write("\"%s\"," % item)
				else:
					f_txt.write("%.2f," % item)
		else:
			for idx,item in enumerate(x):
				f_txt.write("%s," % item)

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

