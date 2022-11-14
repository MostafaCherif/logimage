from tkinter.tix import ROW
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from IPython.display import clear_output
from itertools import combinations
import solveur
from threading import Thread, Event

## Pixelisation

image = 'test.jpg'

img_PIL=Image.open(image)


threshold = 190
i_size = (48,48)
o_size = img_PIL.size

#open file
img=Image.open(image)

#convert to small image
res=img.resize(i_size,Image.BILINEAR)

##resize to output size
#res=small_img.resize(img.size, Image.NEAREST)

#Save output image
name = image[-4]
filename = name+'_{i_size[0]}x{i_size[1]}.jpg'
res.save(filename)

enhancer = ImageEnhance.Contrast(res)
factor = 1.5 #increase contrast
im_output = enhancer.enhance(factor)
im_output.save(filename)


#Display images side by side
plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.title('Original image', size=10)
plt.imshow(img)   #display image
plt.axis('off')   #hide axis
plt.subplot(2,2,2)
plt.title(f'Pixel Art {i_size[0]}x{i_size[1]}', size=10)
plt.imshow(im_output)
plt.axis('off')
#plt.show()

'''im_grayscale = im_output.convert("L")
  
# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
im_edge = im_grayscale.filter(ImageFilter.FIND_EDGES)

im_cropped = im_edge.crop((1, 1, i_size[0] - 1, i_size[1] - 1))
  
im_invert = ImageOps.invert(im_cropped)

im_invert.save('./test/test.jpg')'''

## Transformer l'image en nuances de gris

imgmtplt = mpimg.imread(filename)

R, G, B = imgmtplt[:,:,0], imgmtplt[:,:,1], imgmtplt[:,:,2]
imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B

'''imgGray = mpimg.imread('./test/test.jpg')'''

plt.subplot(2,2,3)
plt.title(f'Pixel Art grayscale', size=10)
plt.imshow(imgGray, cmap='gray')
plt.axis('off')
#plt.show()

## Transformer l'image en noir et blanc avec un seuil (threshold)

imgThreshold = imgGray > threshold

plt.subplot(2,2,4)
plt.title(f'Pixel Art black and white 180', size=10)
plt.imshow(imgThreshold, cmap='gray')
plt.axis('off')
plt.show()

## Image to nonogram list

def ImgToLogimage(im):
    ROW_VALUES = []
    for row in range(i_size[1]):
        ROW_VALUES.append([])
        count = 0
        for el in imgThreshold[row]:
            if el:
                if count != 0:
                    ROW_VALUES[-1].append(count)
                    count = 0
            else :
                count += 1
        if count != 0:
            ROW_VALUES[-1].append(count)

    COL_VALUES = []
    for col in range(i_size[0]):
        count = 0
        COL_VALUES.append([])
        for row in range(i_size[1]):
            if imgThreshold[row][col]:
                if count != 0:
                    COL_VALUES[-1].append(count)
                    count = 0
            else:
                count+=1
        if count != 0:
            COL_VALUES[-1].append(count)

    '''ROW_VALUES.pop(0)
    COL_VALUES.pop(0)
    ROW_VALUES.pop()
    COL_VALUES.pop()'''

# Enlever les colonnes et les lignes vides

    while ROW_VALUES[0] == []:
        ROW_VALUES.pop(0)
    while COL_VALUES[0] == []:
        COL_VALUES.pop(0)
    while ROW_VALUES[-1] == []:
        ROW_VALUES.pop()
    while COL_VALUES[-1] == []:
        COL_VALUES.pop()

    return (ROW_VALUES, COL_VALUES)

# solveur

(ROW_VALUES,COL_VALUES) = ImgToLogimage(imgThreshold)

def thresholdUp():
    global threshold
    threshold += 1
    imgT = imgGray > threshold
    (R, C) = ImgToLogimage(imgT)
    return (threshold,R,C)

# le logimage ainsi créé n'a pas forcément une seule solution, nous allons donc ajouter des cases jusqu'à ce que ça soit le cas

def solvable(r,c):
    s = solveur.NonogramSolver(r,c,"./test")
    return s.solved

def _solvable(r,c):
    action_thread = Thread(target=solvable(r,c))
    action_thread.start()
    action_thread.join(timeout=10)
    s = solveur.NonogramSolver(r,c,"./test")
    return s.solved

def logimage_une_solution():
    global threshold
    global ROW_VALUES
    global COL_VALUES
    if threshold < 256:
        print(threshold)
        if not solvable(ROW_VALUES, COL_VALUES):
            threshold = thresholdUp()[0]
            ROW_VALUES = thresholdUp()[1]
            COL_VALUES = thresholdUp()[2]
            logimage_une_solution()
        else:
            print("Lesgo")
    print("Done")

logimage_une_solution()

#en plus beau

n_col = len(ROW_VALUES)
n_row = len(COL_VALUES)

ROWS = [" ".join(map(str, l)) for l in ROW_VALUES]
COLS = ["\n".join(map(str, l)) for l in COL_VALUES]
cell_text = [[" " for i in range(n_row)] for j in range(n_col)]

print(ROWS)
print(COLS)

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, axs = plt.subplots(1,1)
axs.axis("tight")
axs.axis("off")
the_table = axs.table(cellText = cell_text, rowLabels = ROWS, rowLoc = "right", colLabels = COLS, loc = 'center')
cellDict=the_table.get_celld()
longueurs_col = [len(e) for e in COLS]
max_col = max(longueurs_col)
print(max_col)
for k in range(n_row):
    cellDict[(0,k)].set_height(0.03*max_col)
plt.show()