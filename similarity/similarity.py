import similaripy as sim

def similarity(matrix, k=100, sim_type='cosine', shrink=0, threshold=0, implicit=True, alpha=None, beta=None, l=None, c=None):
    
    # similarity type
    SIM_COSINE      = 'cosine'
    SIM_TVERSKY     = 'tversky'
    SIM_P3ALPHA     = 'p3alpha'
    SIM_ASYMCOSINE  = 'asymcosine'
    SIM_RP3BETA     = 'rp3beta'
    SIM_SPLUS       = 'splus'
    SIM_JACCARD     = 'jaccard'
    SIM_DICE        = 'dice'

    matrix = matrix.T
    

    if sim_type==SIM_COSINE:
        return sim.cosine(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit)
    elif sim_type==SIM_ASYMCOSINE:
        return sim.asymmetric_cosine(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha)
    elif sim_type==SIM_JACCARD:
        return sim.jaccard(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit)
    elif sim_type==SIM_TVERSKY:
        return sim.tversky(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta)
    elif sim_type==SIM_P3ALPHA:
        return sim.p3alpha(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha)
    elif sim_type==SIM_RP3BETA:
        return sim.rp3beta(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta)
    elif sim_type==SIM_SPLUS:
        return sim.s_plus(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, l=l, t1=alpha, t2=beta, c=c)
    elif sim_type==SIM_DICE:
        return sim.dice(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit)
    else:
        print('Error wrong distance metric')






        



        
