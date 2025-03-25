import pandas as pd
from pathlib import Path

EXCEL_PATH = '/Users/AnshulSrivastava/Desktop/IPF/PI Data/Excels'
CSV_PATH = '/Users/AnshulSrivastava/Desktop/IPF/PI Data/CSVs'

# # Load FRIB data
# print('Loading FRIB data...')
# frib = pd.read_excel(f'{FOLDER_PATH}/FRIB-0164.xlsx', sheet_name='E1', skiprows=18, usecols="F,W")
# frib.columns = ['Timestamp', 'Watt Hours Received']

# # Load BCC data
# print('Loading BCC data...')
# bcc = pd.read_excel(f'{FOLDER_PATH}/Business College Complex - 0080.xlsx', sheet_name='E1', skiprows=15, usecols="E,V")
# bcc.columns = ['Timestamp', 'Watt Hours Received']

# # Load Computer Center data
# print('Loading Computer Center data...')
# cc = pd.read_excel(f'{FOLDER_PATH}/Computer Center - 0035.xlsx', sheet_name='E1', skiprows=21, usecols="D,U")
# cc.columns = ['Timestamp', 'Watt Hours Received']

# # Save as CSV
# frib.to_csv(f'{FOLDER_PATH}/FRIB.csv', index=False)
# bcc.to_csv(f'{FOLDER_PATH}/BCC.csv', index=False)
# cc.to_csv(f'{FOLDER_PATH}/CC.csv', index=False)

# # Load BPS Data
# print('Loading BPS data...')
# bps = pd.read_excel(f'{EXCEL_PATH}/Biomedical and Phiscal Science.xlsx', sheet_name='Sheet1', skiprows=20, usecols="E,V")
# bps.columns = ['Timestamp', 'Watt Hours Received']
# bps.to_csv(f'{CSV_PATH}/BPS.csv', index=False)

# # Load Chem Data
# print('Loading Chem data...')
# chem = pd.read_excel(f'{EXCEL_PATH}/Chemistry.xlsx', sheet_name='Sheet1', skiprows=19, usecols="E,V")
# chem.columns = ['Timestamp', 'Watt Hours Received']
# chem.to_csv(f'{CSV_PATH}/Chem.csv', index=False)

# # Load Eng Data
# print('Loading Eng data...')
# eng = pd.read_excel(f'{EXCEL_PATH}/Engineering.xlsx', sheet_name='Sheet1', skiprows=19, usecols="E,V")
# eng.columns = ['Timestamp', 'Watt Hours Received']
# eng.to_csv(f'{CSV_PATH}/Eng.csv', index=False)

# # Load PSS Data
# print('Loading Plant and Soil Sciences data...')
# pss = pd.read_excel(f'{EXCEL_PATH}/Plant and Soil Science.xlsx', sheet_name='Sheet1', skiprows=19, usecols="D,U")
# pss.columns = ['Timestamp', 'Watt Hours Received']
# pss.to_csv(f'{CSV_PATH}/PSS.csv', index=False)

# Common code for all buildings
def load_data(file_name, sheet_name, skiprows, usecol, building_name):
    print(f'Loading {building_name} data...')
    data = pd.read_excel(f'{EXCEL_PATH}/{file_name}', sheet_name=sheet_name, skiprows=skiprows, usecols=usecol)
    data.columns = ['Watt Hours Received']
    # Create custom timestamp column which starts at 1/1/2023 00:00:00 with 1 hour intervals, ends when data ends
    data['Timestamp'] = pd.date_range(start='1/1/2023', periods=len(data), freq='H')
    data.to_csv(f'{CSV_PATH}/{building_name}.csv', index=False)

# Load FRIB data
load_data('FRIB-0164.xlsx', 'E1', 18, 'W', 'FRIB')

# Load BCC data
load_data('Business College Complex - 0080.xlsx', 'E1', 15, 'V', 'BCC')

# Load Computer Center data
load_data('Computer Center - 0035.xlsx', 'E1', 21, 'U', 'CC')

# Load BPS Data
load_data('Biomedical and Phiscal Science.xlsx', 'Sheet1', 20, 'V', 'BPS')

# Load Chem Data
load_data('Chemistry.xlsx', 'Sheet1', 19, 'V', 'Chem')

# Load Eng Data
load_data('Engineering.xlsx', 'Sheet1', 19, 'V', 'Eng')

# Load PSS Data
load_data('Plant and Soil Science.xlsx', 'Sheet1', 19, 'U', 'PSS')

print('Data extraction complete!')