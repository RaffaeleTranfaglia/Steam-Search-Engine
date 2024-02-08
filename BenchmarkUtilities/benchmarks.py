from functools import reduce

'''
    Defines methods to evaluate an IR system through the computation of relevance measures 
    on a set of queries.
'''
class Benchmarks:
    def __init__(self, queries):
        self.queries = queries
    
    '''
        The fraction of the relevant documents (the set R) which has been retrieved.
    '''
    def recall(self, R, A):
        return len(R.intersection(A)) / len(R) if len(R) > 0 else 0
    
    '''
        The fraction of the retrieved documents (the set A) which is relevant.
    '''
    def precision(self, R, A):
        return len(R.intersection(A)) / len(A) if len(A) > 0 else 0
    
    '''
        Precision at Natural Recall Levels.
    '''
    def precisionNRL(self, R, A):
        pNRL = []
        for r in range(1, len(R)+1):
            for a in range(1, len(A)+1):
                pNRL.append((r/len(R), self.precision(R[:r], A[:a])))
        return pNRL
    
    '''
        Precision at Standard Recall Levels.
        Obtained applying ceiling interpolation on the Natural Recall Levels Precision.
    '''
    def precisionSRL(sefl, pNRL):
        i = 0
        pSRL = [(0, pNRL[i][1])]
        for r in range(0.1, 1.1, 0.1):
            if r > pNRL[i][0]:
                i = i + 1
            last = i - 1 if i > 0 else 0
            next = i
            pSRL.append((r, max(pNRL[last][1], pNRL[next][1])))
        return pSRL
    
    '''
        Compute average precision at each standard recall level across all queries examined.
    '''
    def AveragePrecision(self, pSRLs):
        n = len(self.queries)
        if (n != len(pSRLs)):
            print("The number of queries examined does not correspond to the number of precisions.")
            return
        AP = []
        for r in range(0, 1.1, 0.1):
            sum = reduce(lambda x, y : x + y, [p[r][1] for p in pSRLs], 0)
            AP.append((r, sum/n))
        return AP
                