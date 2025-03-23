import numpy as np
from skimage.io import imread , imshow , imsave
import glob



path = '/home/yared/桌面/mackay_data/bc/64934681/ri'
s_p  = '/home/yared/桌面/mackay_data/bc_mfc/cts/144528/64934681/ri'
files = glob.glob(path + '//*.bmp') 
name = []
for i in files:
    n = i.split('/')[-1].split('.')[0].split('_')[-1]
    name.append(n)
name.sort(key=int)   


f = path.split('/')[-2] + '_'  + path.split('/')[-1] + '_'

for i in name[1:len(name)-1]:
    
    img_0 = imread(path + '/' + f + str(int(i)-1) +'.bmp')
    img_1   = imread(path + '/' + f + i +'.bmp')
    img_2 = imread(path + '/' + f + str(int(i)+1) +'.bmp')

    n_img = np.zeros((144,528,3))
    n_img[:,:,0] = img_0
    n_img[:,:,1] = img_1
    n_img[:,:,2] = img_2
    
    imsave(s_p +'/' + f + i + '.bmp',n_img)