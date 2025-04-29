import cv2
import numpy as np
from matplotlib import pyplot as plt
cv2.__version__

img = cv2.imread('./images/audi.jpg')
imgShape = img.shape
print(imgShape)

edgesImg = cv2.Canny(img,100,200)  #Se aplica el filtro Canny con umbrales 100 y 200 y se almacena en edgesImg
plt.subplot(121),plt.imshow(img,cmap = 'gray') #Se hace una subgráfica de la imagen original
plt.title('Imagen original'), plt.xticks([]), plt.yticks([]) #Se establece el título de la imagen y se especifíca que no se ponga información en los ejes
plt.subplot(122),plt.imshow(edgesImg,cmap = 'gray') #Se hace una subgráfica de la imagen filtrada con los bordes resultantes del filtro Canny
plt.title('Imagen filtrada con los bordes'), plt.xticks([]), plt.yticks([]) #Se establece el título de la imagen y se especifíca que no se ponga información en los ejes

plt.show()

cv2.imwrite('./images/audiEdges.jpg',edgesImg)  #Se especifica la ruta y nombre de la imagen. 

