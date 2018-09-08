import h5py
import numpy as np
from PIL import Image
import cv2

hf = h5py.File("test.h5", 'w')

dt = h5py.special_dtype(vlen=np.dtype('uint8'))
g = hf.create_dataset('image', (10,), dtype=dt)

path = "/data1/yang/code/aws/yolo/darknet/data/horses.jpg"
with open(path, "r") as f:
    s = f.read()
im = Image.open(path)
im = np.array(im)

for i in range(10):
    print(i)
    #g[i] = np.fromstring(s, dtype='uint8')
    g[i]=np.fromstring(cv2.imencode(".png", im[:,:,0])[1], dtype=np.uint8)

hf.close()


# then testing t
with h5py.File("test.h5", "r") as f:
    t=f["image"][3]
    arr=cv2.imdecode(t, -1)
    print(arr.shape)
    cv2.imwrite("test.jpg", arr)
