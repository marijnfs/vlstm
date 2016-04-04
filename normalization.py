import os
import numpy as np
#import numpy.ndarray
from matplotlib import pylab as plt
import cv2
import Image
import pdb

pdb.set_trace()
#path = "TrainingData/"
path = "/home/byeon/research/Database/segmetnation/MRBrainS13Data/"
pathr = path+"TrainingData/"
v, w, h = 48,240,240
c = 3


folders = os.listdir(pathr)
n = len(folders)

final = np.zeros((n,v,c,h,w), dtype='d')

for folder in sorted(folders):
    print pathr+folder
    lines = os.listdir(pathr+folder)
    print lines[0][:6]
    files = [ff for ff in lines if  ff == "T1_IR.raw" ] #ff[-3:] == "raw" and ff[0:2] == "T1"
    print len(files)
    nimg = int(folder)-1
    ch = 0
    for fname in sorted(files):
        
        filename = pathr+folder+"/"+fname
        print filename
        values = np.fromfile(filename, dtype='int16')#, sep="")
        print values.shape, v*w*h
        if len(values) == v*w*h:
            volumes = values.reshape([v,h,w])

            for i, img in enumerate(volumes):
                if "5" in folder and i == 0:  
                    temp = volumes[1,:,:]
                    img[img == 0] = temp[img == 0]
                blur = cv2.GaussianBlur(img, (31, 31), 5)
                subtract = (img - blur)
                subtract = subtract.astype(np.float32)
                subtract -= subtract.mean()
                subtract /= subtract.std()
                subtract *= 255. / 4.
                subtract += 128.;
                subtract = np.clip(subtract, 0., 255.)
                subtract = subtract.astype(np.uint8)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
                subtract = clahe.apply(subtract)
                gap = 8
                subtract[:gap,:] = subtract.mean()
                subtract[-gap:,:] = subtract.mean()
                subtract[:,-gap:] = subtract.mean()
                subtract[:,:gap] = subtract.mean()
                print i, subtract.min(), subtract.max(), subtract.mean()
                sub2 = Image.fromarray(subtract)
                fname = folder+"-"+ff[:-3]+str(i)+".png"
                sub2.save("temp/"+fname)

                volumes[i] = subtract.astype(np.int16)
                
            volumes.tofile(open(pathr+folder+"/T1_IR_PP.raw", 'w+'))
#                img /= 255.
#                image = img.astype(np.uint8)
#                image = Image.fromarray(image)
#                fname = folder+"-"+ff[:-3]+str(i)+"-img.png"
#                image.save("temp/"+fname)

            
            
#            plt.imshow(temp)
#            plt.show()


