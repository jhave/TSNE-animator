# TSNE-animator
testing a way to perceive data as fluid animated form!

#Video Output 
https://vimeo.com/192249656/1e988e9e53

#More Info
http://bdp.glia.ca/

# FRAME by FRAME change
simple mod of base TSNE algorithm
so that as parameters change (between values that exceed recommended norms) an imge is saved.

#Example of How-to call thru Terminal: NOTE: change your file source (this demo is set to use Sondheim archive)
python -O TSNE_ARGS.py --learning_rate=1096.1 --exag_rate=66.0 --complexity=2 --perplexity=2 --e_gate_limit=0 --c_gate_limit=1 --lr_gate_limit=0 --learning_rate_INC=0.1 --exag_rate_INC=0.5 --complexity_INC=0.01 --perplexity_INC=0.01 --exag_MAX=88

#
