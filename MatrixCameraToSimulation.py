import numpy as np
from skimage.io import imread
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import LinalgHelper

# get input image files

source_file = "./data"
renderedHDR_fn = join(source_file,"colorchecker_v1_specbrdf.exr")
renderedPNG_fn = join(source_file,"colorchecker_v1_specbrdf.png")
renderedHDR = imread(renderedHDR_fn)
renderPNG = imread(renderedPNG_fn)

cameraHDR_fn = join(source_file,"whiteledcolorchecker1_merged.exr")
cameraPNG_fn = join(source_file,"whiteledcolorchecker.png")
cameraHDR = imread(cameraHDR_fn)
cameraPNG = imread(cameraPNG_fn)

samplePointRender = []
for i in range(4):
    for j in range(6):
        samplePointRender.append((152+111*j,52+111*i))
widthRender = 100
heightRender = 100

samplePointMerge = []
for i in range(4):
    for j in range(6):
        samplePointMerge.append((1189+round(34*1.01*j+0.4*j*j+0.35*i*j),1184+round(37*1.15*i+0.5*i*j)))
widthMerge = 20
heightMerge = 20

# show images

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(renderPNG)
for i in range(len(samplePointRender)):
    ax.add_patch(patches.Rectangle(samplePointRender[i],height=heightRender,width=widthRender,fc="None",ec='r',lw=1))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(cameraPNG)
for i in range(len(samplePointMerge)):
    ax.add_patch(patches.Rectangle(samplePointMerge[i],height=heightMerge,width=widthMerge,fc="None",ec='r',lw=1))
plt.show()

# least squre optimization
dataRender = []
dataMerge = []
for i in range(24):
    x = samplePointRender[i][0]
    y = samplePointRender[i][1]
    dataRender.append(np.average(renderedHDR[y:y+heightRender,x:x+widthRender,:],axis=(0,1)))
    x = samplePointMerge[i][0]
    y = samplePointMerge[i][1]
    dataMerge.append(np.average(cameraHDR[y:y+heightMerge,x:x+widthMerge,:],axis=(0,1)))

dataRender = np.vstack(dataRender)
dataRender = np.hstack([dataRender,np.ones((24,1))])
dataMerge = np.vstack(dataMerge)
dataMerge = np.hstack([dataMerge, np.ones((24,1))])
print(dataRender.shape)
# R @ T = M (N,3) (3,3) = (N,3)
matrixRenderToMerge = np.linalg.pinv(dataRender) @ dataMerge

# white balancing
renderVec = dataRender[18,:]
mergeVec = dataMerge[18,:]
alpha = np.sum(mergeVec)/np.sum(renderVec @ matrixRenderToMerge)

# error
dataConvert = (dataRender @ matrixRenderToMerge)*alpha
error = np.average(np.linalg.norm(dataConvert-dataMerge,axis=0))/np.average(np.linalg.norm(dataMerge,axis=0))
print(f'Least square error is {error}')

# print out some results
for i in range(8):
    x = samplePointRender[i][0]
    y = samplePointRender[i][1]
    renderVec = np.average(renderedHDR[y:y+heightRender,x:x+widthRender,:],axis=(0,1))
    print(f"render vec:{renderVec}")
    x = samplePointMerge[i][0]
    y = samplePointMerge[i][1]
    mergeVec = np.average(cameraHDR[y:y+heightMerge,x:x+widthMerge,:],axis=(0,1))
    print(f"merge vec:{mergeVec}")
    renderConvertVec = np.hstack([renderVec, np.array([1])]) @ matrixRenderToMerge
    print(f"render convert vec:{renderConvertVec}")
    print()
    
