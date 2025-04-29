import cv2
import numpy as np
from matplotlib import pyplot as plt
cv2.__version__

#Leer una imagen

img = cv2.imread('./images/mclaren.jpg')
imgShape = img.shape
print(imgShape)

#Desplegar una imagen

img = img[:,:,::-1]   #Esto porque matplotlib utiliza BGR en vez de RGB
plt.imshow(img)     
plt.show()

#Aplicamos procesamiento de imagen

edgesImg = cv2.Canny(img,100,200)  #Se aplica el filtro Canny con umbrales 100 y 200 y se almacena en edgesImg
plt.subplot(121),plt.imshow(img,cmap = 'gray') #Se hace una subgráfica de la imagen original
plt.title('Imagen original'), plt.xticks([]), plt.yticks([]) #Se establece el título de la imagen y se especifíca que no se ponga información en los ejes
plt.subplot(122),plt.imshow(edgesImg,cmap = 'gray') #Se hace una subgráfica de la imagen filtrada con los bordes resultantes del filtro Canny
plt.title('Imagen filtrada con los bordes'), plt.xticks([]), plt.yticks([]) #Se establece el título de la imagen y se especifíca que no se ponga información en los ejes

plt.show()

#Guardar imagen

cv2.imwrite('./images/mclarenEdges.jpg',edgesImg)  #Se especifica la ruta y nombre de la imagen. 

img2 = cv2.imread('./images/lambo.jpg')  #Se lee la imagen
img2Shape = img2.shape                   #Se obtiene su forma
print("Forma de la imagen original= ",img2Shape)  #Se imprime su forma
img2 = img2[:,:,::-1]   #Esto porque matplotlib utiliza BGR en vez de RGB
plt.imshow(img2)     #Se grafica 
plt.show()

img2Resized = cv2.resize(img2, (1920, 1280)) #Se modifica 
img2ResizedShape = img2Resized.shape
print("Forma de la imagen despues del resize = ",img2ResizedShape)
plt.imshow(img2Resized)     
plt.show()

#imagenMezclada = cv2.add(img,img2Resized)
imagenMezclada = cv2.addWeighted(img,0.7,img2Resized,0.3,0)
plt.imshow(imagenMezclada)     
plt.show()