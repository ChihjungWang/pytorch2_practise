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
                if duration.days > 7:
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
img2img_path = 'D:/AI_master/github/stable-diffusion-webui/outputs/img2img-images'
img2img_grids_path = 'D:/AI_master/github/stable-diffusion-webui/outputs/img2img-grids'
txt2img_path = 'D:/AI_master/github/stable-diffusion-webui/outputs/txt2img-images'
txt2img_grids_path = 'D:/AI_master/github/stable-diffusion-webui/outputs/txt2img-grids'


file_type = ['.jpg','.png']
days = 7

clearTempFiles(img2img_path, file_type, days)
clearEmptyFolders(img2img_path)

clearTempFiles(img2img_grids_path, file_type, days)
clearEmptyFolders(img2img_grids_path)

clearTempFiles(txt2img_path, file_type, days)
clearEmptyFolders(txt2img_path)

clearTempFiles(txt2img_grids_path, file_type, days)
clearEmptyFolders(txt2img_grids_path)