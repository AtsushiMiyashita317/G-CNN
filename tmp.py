import pandas as pd

x = pd.DataFrame(index=[],columns=['a','b'])
print(min(-8,x['a'].min()))
print(max(10,x['a'].max()))
