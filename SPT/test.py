# import numpy as np
# x = np.array([[1,1,np.nan],[4,2,np.nan],[2,3,np.nan]])
# # print(x)
# # print(np.nanstd(x, axis=0))
# # print(x ** 2)
# # print(np.sum(x ** 2, axis=0))
# # result = np.where(np.isnan(x).all(axis=0), np.nan, np.nansum(x, axis=0))
# # print(result)
# print(5 < np.inf)
car = {
  "1.5": "Ford",
  "3": "Mustang",
  "2": 1964
}

x = car.items()

print(x)

y= sorted(car.items(), key = lambda x: float(x[0]))

print(y)
