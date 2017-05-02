###This Script Require Python 2.7###

from xlrd import open_workbook
import xlsxwriter
workbook = xlsxwriter.Workbook('SLROBJ.xlsx')
from practnlptools.tools import Annotator
import re
annotator=Annotator()
data_test = wb = open_workbook('../data/trainStance.xlsx')
for sheet in wb.sheets():
    number_of_rows = sheet.nrows
    number_of_columns = sheet.ncols
    write_dict=dict()
    items = []
    final_str = list() 
    rows = []
    for row in range(1, number_of_rows):
        values = []
        for col in range(number_of_columns):
            value  = (sheet.cell(row,col).value)
            try:
                value = str(int(value))
            except ValueError:
                pass
            finally:
                values.append(value)
        line_clean=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",values[2]).split())
        v1 = annotator.getAnnotations(line_clean)['srl']
        ff = " "
        for ele in v1:
            try:
                a = ele['A1']
                ff = ff + a + " " 
            except KeyError, e:
                try:
                    a = ele['V']
                    ff = ff + a + " " 
                except KeyError, e:
                    print("************")
                    print(ele)
                    continue
                #print("Caught A1 warning! Ignoring")
            #final_str.append(a)
        print("Running for ", row)
        print(ff)
        write_dict[row]=ff

row = 0
col = 0
worksheet = workbook.add_worksheet()
for key in write_dict.keys():
    row += 1
    worksheet.write(row, col, key)
    item = write_dict[key]
    worksheet.write(row, col + 1, item)
workbook.close()
print(write_dict)