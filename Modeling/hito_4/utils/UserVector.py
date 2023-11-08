import pandas as pd

class UserVector:
    def __init__(self,taxnumberprovider:str, df:pd.DataFrame,length:int = 10) -> None:
         
        self.df = df 
        self.id = taxnumberprovider
        #TODO add randomizer if > 10 descriptions
        self.strings = self.df.query(f"taxnumberprovider == '{str(taxnumberprovider)}'")['feature_vector'].unique()[:length]
        self.strings = [ i.replace("\n",' ') for i in self.strings]
         
        #TODO revisar que el df no sea nulo
        #if self.df == None:
        #    print('This user has no AgileBuys.')
         
        #print(self.strings)
        #TODO crear una funcion que describa al usuario
        