import numpy as np
import pandas as pd
pd.__version__

pd.<TAB>
pd.read<TAB>

# Pandas Series is a one-dimensional array of indexed data

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
data.values
data.index
data[1]
data[1:3]

# can explicitly define index for Series object
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data
data['b']

#Series directly from dictionary

population_dict = {'California': 38332521,
'Texas': 26448193,
'New York': 19651127,
'Florida': 19552860,
'Illinois': 12882135}
population = pd.Series(population_dict)
population
population['California']
population['California':'Illinois']

#can behave how you want through key argument
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])


#DataFrame
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area
states = pd.DataFrame({'population': population,
'area': area})
states
states.index
states.columns

states['area']

pd.DataFrame(population, columns=['population'])

data = [{'a': i, 'b': 2 * i} for i in range(3)]
pd.DataFrame(data)

pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

#dictionary of Series objects
pd.DataFrame({'population': population, 'area': area})

#from a 2-d NumPy array
pd.DataFrame(np.random.rand(3, 2),
columns=['foo', 'bar'],
index=['a', 'b', 'c'])

#Index as immutable array or ordered set
ind = pd.Index([2, 3, 5, 7, 11])
ind
print(ind.size, ind.shape, ind.ndim, ind.dtype)
ind[1] = 0

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB #intersection
indA | indB #union
indA ^ indB #symmetric difference


#data selection of Series
data = pd.Series([0.25, 0.5, 0.75, 1.0],
index=['a', 'b', 'c', 'd'])
data
data['b']
'a' in data
data.keys()
list(data.items())

data['e'] = 1.25
data
data['a':'c']
data[0:2]
data[(data > 0.3) & (data < 0.8)]
data[['a', 'e']]

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data
data[1]
data[1:3]

#reference the explicit index
data.loc[1]
data.loc[1:3]

#reference the implicit index
data.iloc[1]
data.iloc[1:3]


#data selection of DataFrame
area = pd.Series({'California': 423967, 'Texas': 695662,
'New York': 141297, 'Florida': 170312,
'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
'New York': 19651127, 'Florida': 19552860,
'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data

data['area']
data.area

data.area is data['area']
data.pop is data['pop']

data['density'] = data['pop'] / data['area']
data

data.values

data.T
data.values[0]
#vs
data['area']

data.iloc[:3, :2]
data.loc[:'Illinois', :'pop']
#hybrid
data.ix[:3, :'pop']

data.loc[data.density > 100, ['pop', 'density']]

#any indexing convention can be used to set values
data.iloc[0, 2] = 90
data

#indexing refers to columns, slicing refers to rows
data['Florida':'Illinois']

#direct masking operations are also interpreted row-wise
data[data.density > 100]


#Ufuncs index preservation
df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
columns=['A', 'B', 'C', 'D'])
df

#indices preserved
np.exp(df)
np.sin(df * np.pi / 4)

#index alignment
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
'New York': 19651127}, name='population')
population / area

area.index | population.index

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B

A.add(B)
A.add(B, fill_value = 0)

A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
columns=list('AB'))
A
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
columns=list('BAC'))
B
A + B

fill = A.stack().mean()
A.add(B, fill_value=fill)

#operations
A = rng.randint(10, size=(3, 4))
A
A - A[0]

#operations rowwise by default
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]
#columnwise
df.subtract(df['R'], axis=0)

halfrow = df.iloc[0, ::2]
halfrow

df - halfrow

#class 2

import numpy as np
import pandas as pd

vals1 = np.array([1, None, 3, 4])
vals1

for dtype in ['object', 'int']:
    print("dtype =", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()
    
vals1.sum()

vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype

1 + np.nan
0 * np.nan

vals2.sum(), vals2.min(), vals2.max()

np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

#upcasting
pd.Series([1, np.nan, 2, None])

x = pd.Series(range(2), dtype=int)
x

x[0] = None
x

data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
#masking
data[data.notnull()]

data.dropna()
df = pd.DataFrame([[1, np.nan, 2],
[2, 3, 5],
[np.nan, 4, 6]])
df

df.dropna()
df.dropna(axis='columns')
df[3] = np.nan
df
df.dropna(axis='columns', how='all')
df.dropna(axis='rows', thresh=3)
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)

#forward-fill to propogate previous values forward
data.fillna(method='ffill')

#back-fill
data.fillna(method='bfill')

#along rows
df.fillna(method='ffill', axis=1)


#hierarchical indexing
index = [('California', 2000), ('California', 2010),
('New York', 2000), ('New York', 2010),
('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
18976457, 19378102,
20851820, 25145561]
pop = pd.Series(populations, index=index)
pop

pop[('California', 2010):('Texas', 2000)]

pop[[i for i in pop.index if i[1] == 2010]] #not as efficient as slicing

#Pandas MultiIndex
index = pd.MultiIndex.from_tuples(index)
index

pop = pop.reindex(index)
pop

pop[:, 2010] #much more efficient

pop_df = pop.unstack()
pop_df

pop_df.stack()

pop_df = pd.DataFrame({'total': pop,
'under18': [9267089, 9284094,
4687374, 4318033,
5906301, 6879014]})
pop_df

f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()


















