'''
    Defines methods to evaluate an IR system through the computation of relevance measures.
'''
class Measures:
    def __init__(self):
        pass
    
    '''
        The fraction of the relevant documents (the set R) which has been retrieved.
    '''
    @staticmethod
    def recall(RA, R):
        return RA / R
    
    '''
        The fraction of the retrieved documents (the set A) which is relevant.
    '''
    @staticmethod
    def precision(RA, A):
        return RA / A
    
    
            