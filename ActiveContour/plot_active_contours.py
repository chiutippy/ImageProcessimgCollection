import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.io import imread,imsave
import time

###record start time###
start_time = time.time()

def ac(image,init):
    
    snake = active_contour(
        gaussian(image, sigma=3, preserve_range=False),
        init,
        alpha=0.015,
        beta=10,
        gamma=0.001,
    )
    
    return snake

def show_imag(image,result):
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(initial[:, 1], initial[:, 0], '--r', lw=3)
    ax.plot(result[:, 1], result[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()
         
'''set initial contour'''

s = np.linspace(0, 2 * np.pi, 400)
r = 50 + 60 * np.sin(s)#x axis, bigger will go down + y range, bigger will go longer
c = 300 + 70 * np.cos(s)
initial = np.array([r, c]).T    

'''main'''

save_path = r'\active_contour_results'

for i in range(0,49):
    
    img = imread(r'data\images'+'/'+ str(i) +'.bmp' )    
    contour = ac(img,initial).astype(np.int32)
    
    ##put contour on new image
    m,n = img[:,:,0].shape
    n_img = np.zeros((m,n,3), dtype=np.uint8)    
    r,c = draw.polygon(contour[:,0], contour[:,1])
    draw.set_color(n_img, [r,c], [255,255,255])
    
    imsave(save_path+'/'+str(i)+'.bmp', n_img)
    show_imag(img, contour)

###estimate run time###
end_time = time.time()
execution_time = end_time - start_time
print("runtime：%.3f" % execution_time, "s")
