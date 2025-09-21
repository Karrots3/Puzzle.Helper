from types import List
import numpy as np

class Edge:
    def Edge(self,margin:np.ndarray):
        self.margin = margin
        
    

class Piece:
    def Piece(self, edges:List[Edge]):
        assert len(edges)==4
        self.edges = edges
        self.top:Piece= None
        self.bottom:Piece= None
        self.left:Piece = None
        self.right:Piece = None
        self.is_margin=False


    
def pieces_score(pa:Piece, pb:Piece, method)







    