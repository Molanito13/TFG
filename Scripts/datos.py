import skimage
import os
import numpy as np
array = []
os.chdir("..")
with os.scandir('./archive_grande/ASL_Dataset/Train') as f:
    for aux in f:
        with os.scandir(aux.path) as g:
            for aux2 in g:
                #print(aux2.path)
                img = skimage.io.imread(aux2.path)
                if img.shape != (70, 70, 3):
                    print(aux2.path)
                    continue

                array.append(img) #HE QUITADO EL .flatten()


array = np.array(array)

print(len(array)) #Segundo pixel, valores RGB Una sola foto son 480K iteraciones para sacar el RGB de cada pixel de una iamgen de 400x400
# Todo el Abecedario tarda 40.373.760.000 iteraciones que equivale a 112 horas de ejecucion

np.save('DatosTrainCONV', array)