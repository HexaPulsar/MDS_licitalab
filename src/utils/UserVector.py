import pandas as pd
import numpy as np
class UserVector:
    def __init__(self,taxnumberprovider:str, 
                 df:pd.DataFrame,
                 length:int = 10) -> None:
          
        df = df.query(f"taxnumberprovider == '{str(taxnumberprovider)}'")['feature_vector'].unique()
        self.strings =np.random.choice(df , size=length, replace=False) 
        self.strings = [ i.replace("\n",' ') for i in self.strings]
    
        