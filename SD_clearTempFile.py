import os, shutil
from datetime import datetime

def clearTempFiles(dir_path,ext_type,days):
    currentDateAndTime = datetime.now()
    del_image_list = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1] in ext_type:
                fullpath = os.path.join(root, f)
                create_time = os.path.getmtime(fullpath)
                create_time = datetime.fromtimestamp(create_time)
                duration = currentDateAndTime - create_time
                if duration.days > days:
                    os.remove(fullpath)
                    print(f'delete >>> {fullpath}')

def clearEmptyFolders(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        if os.listdir(root) == ['Thumbs.db']:
            shutil.rmtree(root)
            print(f'delete >>> {root}')
        elif not os.listdir(root):
            os.rmdir(root)
            print(f'delete >>> {root}')
        # print(f'root:{root}')
        # print(f'dirs:{dirs}')


# dir_path = 'F:/AI_master/github/stable-diffusion-webui/outputs/python_tools_test/img2img-images'

root_path = '//mcd-one/ta/11_AI_Tools/stable-diffusion/stable_diffusion_output'
PC_list = ['CY0010117','CY0010120','CY0012297','CY-CS-01458']
folder_list = ['img2img-images','img2img-grids','txt2img-images','txt2img-grids']


file_type = ['.jpg','.png']
days = 14

for PC in PC_list:
    for folder in folder_list:
        dir_path = os.path.join(root_path,PC,'outputs',folder)
        clearTempFiles(dir_path, file_type, days)
        clearEmptyFolders(dir_path)