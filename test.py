import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
print(cur_dir)
workbook = pd.read_excel(cur_dir + "\Models_Information.xlsx", sheet_name=0)

headers = [col for col in workbook.columns]
print(len(headers))

for i in range(len(workbook)):
    print()
#     for header in headers:
#         print(type(workbook.loc[i][header]))

