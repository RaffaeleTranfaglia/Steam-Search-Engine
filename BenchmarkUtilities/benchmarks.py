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
        return len(set(R).intersection(set(A))) / len(R) if len(R) > 0 else 0
    
    '''
        The fraction of the retrieved documents (the set A) which is relevant.
    '''
    def precision(self, R, A):
        return len(set(R).intersection(set(A))) / len(A) if len(A) > 0 else 0
    
    '''
        Precision at Natural Recall Levels.
    '''
    def precisionNRL(self, R, A):
        pNRL = []
        r = 1
        for a in range(len(A)):
            if A[a] in R:
                pNRL.append((r/len(R), r/(a+1)))
                r = r + 1
        for i in range(r, len(R)+1):
            pNRL.append((i/len(R), 0))
        return pNRL
    
    '''
        Precision at Standard Recall Levels.
        Obtained applying ceiling interpolation on the Natural Recall Levels Precision.
    '''
    def precisionSRL(sefl, pNRL):
        i = 0
        pSRL = [(0, pNRL[i][1])]
        for r in range(1, 11):
            if (i+1 < len(pNRL) and r/10 > pNRL[i][0]):
                print(r/10)
                print(pNRL[i][0])
                i = i + 1
            last = i - 1 if i > 0 else 0
            next = i
            pSRL.append((r/10, max(pNRL[last][1], pNRL[next][1])))
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
        for r in range(0, 11):
            sum = reduce(lambda x, y : x + y, [p[r][1] for p in pSRLs], 0)
            AP.append((r/10, sum/n))
        return AP

# test
if __name__ == "__main__":
    q = [{
        "UIN" : "Simulator of real farms",
        "query" : "farm simulator",
        "fields" : ["name", "description"],
        "relevant_documents" :
        [
            236790, 259890, 273790, 92900, 313160, 447020, 90200, 787860, 612030, 220260
        ]
    },
    {
        "UIN" : "Planes flight simulator in virtual reality",
        "query" : "flight simulator vr",
        "fields" : ["name", "description", "cgt"],
        "relevant_documents" :
        [
            314160, 269950, 434030, 223750, 1001240, 347170, 862390
        ]
    }]
    b = Benchmarks(q)
    a = [0, 314160, 1, 269950, 434030, 3, 347170, 4, 5, 862390, 223750]
    pNRL = b.precisionNRL(b.queries[1]["relevant_documents"], a)
    pSRL = b.precisionSRL(pNRL)
    print(pNRL)
    print(pSRL)
                