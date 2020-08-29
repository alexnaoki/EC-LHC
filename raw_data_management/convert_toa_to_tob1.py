import os
from pathlib import Path

# Program to be used
# toa_to_tob1_path = Path(r"C:\Program Files (x86)\Campbellsci\LoggerNet\toa_to_tob1.exe")
toa_to_tob1_path = r"C:\Program Files (x86)\Campbellsci\LoggerNet\toa_to_tob1"

# Path of the folder containing TOA5 files
toa5_path = Path(r'E:\Teste_fase03_result\toa5_Mergecopy')

# List of TOA5 files
toa5_files = toa5_path.glob('*.dat')

# Creation of a new folder for TOB1 files
folder_name = 'CONVERTED_TO_TOB1_FASE03'

try:
    os.mkdir(Path.joinpath(toa5_path.parent, folder_name))
except:
    pass

# print("Files to be converted: {}".format([i.name for i in toa5_files]))


for i in toa5_files:
    print('Converting:')
    print(i)
    print()

    path_folder = Path.joinpath(i.parents[1], folder_name)
    # print(path_folder)

    tob1_file = 'TOB1_{}'.format(i.name[5:])
    # print(tob1_file)

    tob1_path = path_folder / tob1_file
    print(tob1_path)

    print(toa_to_tob1_path)
    os.system('"{}" {} {}'.format(toa_to_tob1_path, i, tob1_path))
