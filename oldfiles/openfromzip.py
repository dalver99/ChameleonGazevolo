import os
from zipfile import ZipFile
from PIL import Image
from io import BytesIO

def import_images_from_zip(zip_file_name):
    with ZipFile(zip_file_name, 'r') as zip:
        file_names = zip.namelist()
        images = []
        for file_name in file_names:
            if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                data = zip.read(file_name)
                image = Image.open(BytesIO(data))
                images.append(image)
    return images

zip_file_name = 'frames.zip'
images = import_images_from_zip(zip_file_name)
#잘 임포트 됐는지 확인
print(len(images))