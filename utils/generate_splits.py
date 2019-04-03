import os


images_path = '/SSD/fashion-mnist/images/'
image_list = os.listdir(images_path)

with open('/SSD/fashion-mnist/trainlist.txt','w') as f:
    for i in range(0,15000):
        f.write(images_path+str(image_list[i])+'\n')
    f.close()

with open('/SSD/fashion-mnist/testlist.txt','w') as f:
    for i in range(15000,17500):
        f.write(images_path+str(image_list[i])+'\n')
    f.close()