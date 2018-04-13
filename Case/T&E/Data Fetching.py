import os
import cx_Oracle
import csv
import pandas as pd

# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_SINGAPORE.ZHT16GBK'
# os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.WE8MSWIN1252'
os.environ['NLS_LANG'] = 'ENGLISH_UNITED KINGDOM.AL32UTF8'

# You can set these in system variables but just in case you didnt
# os.putenv('ORACLE_HOME', '/oracle/product/10.2.0/db_1')
# os.putenv('LD_LIBRARY_PATH', '/oracle/product/10.2.0/db_1/lib')
os.putenv('ORACLE_HOME', "C:\\app\oracle\\client11g\\product\\11.2.0\\client_2") # 环境变量，需要设置双斜杠\\
os.putenv('LD_LIBRARY_PATH', "C:\\app\\oracle\\client11g\\product\\11.2.0\\client_2\\lib")

# connection = cx_Oracle.connect('APAC_CONCUR_CORE/not_apac_concur@exak-scan.kau.roche.com:15210/OTHER_GODWP.KAU.ROCHE.COM')

dsn_tns = cx_Oracle.makedsn('exak-scan.kau.roche.com', 15210, service_name='OTHER_GODWP.KAU.ROCHE.COM')
print(dsn_tns)
connection = cx_Oracle.connect('APAC_CONCUR_CORE', 'not_apac_concur', dsn_tns)

SQL = "select count(1) from co_workflow_cycle fetch first 10 ROWS ONLY"

# multiline Sql statement
SQL = """
        SELECT count(1)
           FROM co_workflow_cycle
           WHERE 
           CREATE_DATE >=to_date('2017.11.01 00:00:00',  'YYYY.MM.DD HH24:MI:SS')
           AND (COMPANY_CODE = 1753 
           OR COMPANY_CODE = 1760 )
           AND COUNTRY_CODE = 'CN'
"""

# Use cursor, Write to file csv
'''
# Network drive somewhere
filename = ".\Output.csv"
FILE = open(filename, "w");
output = csv.writer(FILE, dialect='excel')

cursor = connection.cursor()
cursor.execute(SQL)
for row in cursor:
    output.writerow(row)
cursor.close()
connection.close()
FILE.close()
'''

# get a list from fetchall()
"""
SQL = "select distinct(report_date) from DM_F_YF_CM_SRC where COMPANY_CODE = '1753' order by REPORT_DATE"
cursor = connection.cursor()
cursor.execute(SQL)
result = cursor.fetchall()
cursor.close()
print(result)
"""

# Load data from orable to pandas dataframe
df_ora = pd.read_sql(SQL, con=connection)
print(df_ora)
connection.close()
