from PIL import Image
import skimage
import os

j = 0
os.chdir("..")
with os.scandir('./archive_grande/ASL_Dataset/Test') as f:
    for aux in f:
        with os.scandir(aux.path) as g:
            for aux2 in g:
                print(aux2.path)
                imagen_original = Image.open(aux2.path)
                imagen_redimensionada = imagen_original.resize((70, 70))
                imagen_redimensionada.save(aux2.path)
                j+=1

print(j)

