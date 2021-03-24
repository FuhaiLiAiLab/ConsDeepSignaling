from statsmodels.distributions.empirical_distribution import ECDF

permuted_pearson = [-0.0228598, -0.0226568, -0.0217334, 0.0122412, 0.0178914,
            0.0141566, -0.0038268, 0.0099058, -0.0370762, -0.0008438]

ecdf = ECDF(permuted_pearson)

# P-value for 5-fold Sets
print('P(x>0.850293): %.3f' % (1.0 - ecdf(0.850293)) )
print('P(x>0.842299): %.3f' % (1.0 - ecdf(0.842299)) )
print('P(x>0.855838): %.3f' % (1.0 - ecdf(0.855838)) )
print('P(x>0.866354): %.3f' % (1.0 - ecdf(0.866354)) )
print('P(x>0.859415): %.3f' % (1.0 - ecdf(0.859415)) )
# P-value for Average 
print('P(x>0.8548398): %.3f' % (1.0 - ecdf(0.8548398)) )

import rpy2.robjects as R

pvalue_list = [0.000, 0.000, 0.000, 0.000, 0.000] # my pvalues
p_adjust = R['p.adjust'](R.FloatVector(pvalue_list),method='BH')
for v in p_adjust:
    print(v)