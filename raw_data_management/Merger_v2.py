import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pathlib

class Merger_TOA5():
    """The objective of this program is to merge same day TOA5 data.

    Parameters
    ----------
    path : string
        The path is the main folder containing subfolders of TOA5 type files.

    Attributes
    ----------
    path

    """
    def __init__(self, path):
        self.path = pathlib.Path(path)
        print('Path: {}'.format(self.path))

    def checking_files(self, lower_limit=200000000):
        # List all files and its path from all subfolders
        self.full_path_files = []
        self.files = []
        # print(os.listdir(self.path))

        for root, directory, file in os.walk(self.path):
            for f in file:
                if '.dat' in f:
                    self.full_path_files.append(os.path.join(root, f))
                    self.files.append(f)
                else:
                    pass

        # Check files with lower size than a lower limit
        self.file_size = []
        print('POSSIBLE INCOMPLETE FILES: ')
        for p, f in zip(self.full_path_files, self.files):
            if ('TOA5' in f) and ('flux' not in f) and ('log' not in f) and (
                    os.path.getsize(p) < lower_limit):
                print(f, os.path.getsize(p) / 1000, 'KB')
                self.file_size.append(os.path.getsize(p) / 1000)

            else:
                pass

        self.NOT_incomplete = []
        check = input(
            "\nPress 'Yes' to confirm incomplete files\nOtherwise write any other Key the file to be excluded from the Merger process\n"
        )
        while (check != 'Yes'):
            check = input("Insert File name: ")
            print("Write 'Yes' to exit.")
            if check == 'Yes':
                pass
            else:
                self.NOT_incomplete.append(check)
        print("Files Removed from list:\n{}".format(self.NOT_incomplete))

        # List incomplete files from TOA5
        self.incomplete_files_TOA5 = []
        for p, f in zip(self.full_path_files, self.files):
            if ('TOA5' in f) and ('flux' not in f) and ('log' not in f) and (
                    os.path.getsize(p) <
                    lower_limit) and (f not in self.NOT_incomplete):
                self.incomplete_files_TOA5.append([f, os.path.getsize(p)])
            else:
                pass
        print("\nList of Incomplete Files:")
        for i in self.incomplete_files_TOA5:
            print(i[0], i[1] / 1000, "KB")

    def identify_sameDay(self):
        # file_number = [int(column[0][19:-4]) for column in self.incomplete_files_TOA5[:]]
        file_number = [int(column[0][19:-20]) for column in self.incomplete_files_TOA5[:]]
        file_size = [int(column[1]) for column in self.incomplete_files_TOA5[:]]
        self.merge = []
        for i in range(len(self.incomplete_files_TOA5)):
            proximo = file_number[i] + 1
            anterior = file_number[i] - 1
            if proximo in file_number:
                print("Merge {} and {}".format(file_number[i], proximo), sum([file_size[i], file_size[i+1]]))
                self.merge.append(file_number[i])
            elif (proximo not in file_number) & (anterior in file_number):
                pass
            else:
                print('File without complement')
        print(self.merge)
        # return merge

    def merge_sameDay(self, path, name_folder):
        # Copy files
        os.mkdir(r'{}\\{}'.format(path, name_folder))
        path_folder = os.path.join(path, name_folder)
        self.files_toCopy = []
        for merge01 in self.merge:
            merge02 = merge01 + 1
            for file in self.incomplete_files_TOA5:
                print(file[0][19:-4])
                if ('TOA5' in file[0]) and ('flux' not in file[0]) and ('log' not in file[0]):
                    # if merge01 == int(file[0][19:-4]):
                    if merge01 == int(file[0][19:-20]):
                        self.files_toCopy.append(file[0])
                    # if merge02 == int(file[0][19:-4]):
                    if merge02 == int(file[0][19:-20]):
                        self.files_toCopy.append(file[0])

        for p, file in zip(self.full_path_files, self.files):
            if file in self.files_toCopy:
                shutil.copyfile(p, os.path.join(path_folder, file))
                print('To MERGE')
                print(p)
            else:
                if ('TOA5' in file) and ('flux' not in file) and ('log' not in file):
                    print(file)
                    # shutil.copyfile(p, os.path.join(path_folder, file))
                else:
                    pass


        # Merge phase
        for merge01 in self.merge:
            merge02 = merge01 + 1
            for i in range(len(self.files_toCopy)):
                # if merge01 == int(self.files_toCopy[i][19:-4]):
                if merge01 == int(self.files_toCopy[i][19:-20]):
                    print("Merging... {} and {}".format(self.files_toCopy[i], self.files_toCopy[i+1]))
                    f1 = open(os.path.join(path_folder, self.files_toCopy[i])).readlines()
                    f2 = open(os.path.join(path_folder, self.files_toCopy[i+1])).readlines()
                    with open(os.path.join(path_folder, self.files_toCopy[i]),'w') as output:
                        for j in range(len(f1)):
                            output.write(str(f1[j]))
                        for j in range(4, len(f2)):
                            output.write(str(f2[j]))
                    os.remove(os.path.join(path_folder, self.files_toCopy[i+1]))
                    print("Merged: {} and {}".format(self.files_toCopy[i], self.files_toCopy[i+1]))

    def copy_full_tob1(self, path, name_folder, lower_limit=90000000):
        try:
            # os.mkdir('{}'.format(path_to_copy_TOB1))
            os.mkdir(r'{}\\{}'.format(path, name_folder))
            path_folder = os.path.join(path, name_folder)
        except:
            pass
        print('Copying TOB1 files:')
        for root, directory, file in os.walk(self.path):
            # print(root)
            for f in file:
                full_path_f = os.path.join(root, f)
                if ('.dat' in f) and ('TOB1' in f) and ('flux' not in f) and ('log' not in f) and (os.path.getsize(full_path_f)>lower_limit):
                    print(full_path_f)
                    shutil.copyfile(full_path_f, os.path.join(path_folder, f))
