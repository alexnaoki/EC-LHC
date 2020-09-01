import shutil
import pathlib
import os

class Merger_TOA5():
    def __init__(self, path):
        self.path = pathlib.Path(path)
        print(f'Path:{self.path}')

    def identifyandmerge_sameday(self):
        # Only TOA5 files
        files = self.path.rglob('TOA5_11341.ts_data*.dat')
        min_fileSize = 200000000

        # Create merge folder
        # print(self.path.parents[0])
        self.merge_folder = self.path.parents[0]/'merge'
        self.merge_folder.mkdir(exist_ok=True)

        # List files below threshold size
        self.incomplete_files = []
        self.file_number = []
        for file in files:
            if file.stat().st_size < min_fileSize:
                if pathlib.Path(self.merge_folder/file.name).is_file():
                    print('file exist')
                else:
                # print(files[i])
                # print(file.name[19:-20])
                    print(file.stat().st_size)
                    self.incomplete_files.append(file)
                    self.file_number.append(int(file.name[19:-20]))

        # Copying files to merge
        self.files_to_merge = []
        self.files_to_merge_number = []
        for file, number in zip(self.incomplete_files, self.file_number):
            if (number+1 in self.file_number) or (number-1 in self.file_number):
                print(file)
                self.files_to_merge.append(self.merge_folder/file.name)
                self.files_to_merge_number.append(number)
                shutil.copyfile(src=file, dst=self.merge_folder/file.name)

        # print(self.files_to_merge_number)
        # print(self.files_to_merge)

        for merge01 in self.files_to_merge_number:
            merge02 = merge01 + 1
            # print('fadsf')
            for i in range(len(self.files_to_merge)):
                # print(merge01)
                if merge02 == int(self.files_to_merge[i].name[19:-20]):
                    print(f'Merging... {self.files_to_merge[i-1]} and {self.files_to_merge[i]}')
                    f1 = open(self.files_to_merge[i-1]).readlines()
                    f2 = open(self.files_to_merge[i]).readlines()
                    with open(self.files_to_merge[i-1], 'w') as output:
                        for j in range(len(f1)):
                            output.write(str(f1[j]))
                        for j in range(4, len(f2)):
                            output.write(str(f2[j]))
                    self.files_to_merge[i].unlink()
                    print(f'Deleting {self.files_to_merge[i]}')

    def copy_tob_files(self):
        # Only TOB1 files
        files = self.path.rglob('TOB1_11341.ts_data*.dat')
        min_fileSize = 100000000

        # Create merge folder
        # print(self.path.parents[0])
        self.complete_folder = self.path.parents[0]/'tob1_complete'
        self.complete_folder.mkdir(exist_ok=True)

        self.complete_files = []
        for file in files:
            if file.stat().st_size >= min_fileSize:
                print(file)
                if pathlib.Path(self.complete_folder/file.name).is_file():
                    print('file exist')

                else:
                    self.complete_files.append(file)
                    shutil.copyfile(src=file, dst=self.complete_folder/file.name)
        # copyAll = input('Copy all files ? ')
        # print(type(copyAll))

    def convert_toa_to_tob1(self, path_toa_to_tob1):
        # path_toa_to_tob1 path_toa_to_tob1th
        toa5_files_to_convert = self.merge_folder.rglob('TOA5*.dat')
        for toa5_file in toa5_files_to_convert:
            # if toa5_file.is_file():
            #     print('file exists')
            # else:
            print('Converting: ',toa5_file)

            tob1_path = toa5_file.parents[0]
            # print(tob1_path)

            tob1_file = f'TOB1_{toa5_file.name[5:]}'
            # print(tob1_file)

            tob1_pathfile = tob1_path/tob1_file
            if pathlib.Path(tob1_pathfile).is_file():
                print('ja existe')
            else:

                os.system(f'"{path_toa_to_tob1}" {toa5_file} {tob1_pathfile}')

    def join_mergedFiles_fullFiles(self):
        tob1_files_merged = self.merge_folder.rglob('TOB1*.dat')
        for file in tob1_files_merged:
            if pathlib.Path(self.complete_folder/file.name).is_file():
                print('ja existe')
            else:
            # if file.is_file():
            #     print('file exist')
            # else:
                print('Copied:', file)
                shutil.copyfile(src=file, dst=self.complete_folder/file.name)


# Main folder
# path_teste = r'G:\Meu Drive\USP-SHS\Exemplo_apagar_dps'
path_teste = r'E:\Teste_merger_func'
# Start program
a = Merger_TOA5(path=path_teste)

# Identify and Merge data from same day
a.identifyandmerge_sameday()

# Copy Tob1 files to newfolder
a.copy_tob_files()

# Convert TOA5 files to tob1 using merge folder
a.convert_toa_to_tob1(path_toa_to_tob1=r"C:\Program Files (x86)\Campbellsci\LoggerNet\toa_to_tob1")

# Copy converted TOB1 files to tob1_complete folder
a.join_mergedFiles_fullFiles()
