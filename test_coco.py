import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
annType = ['segm','bbox','keypoints']
annType = annType[0]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.', annType)
#initialize COCO ground truth api
dataDir='../'
dataType='val2014'
annFile = '/root/data/TongueSegForMaskRCNN_FB_1024x768/coco/annotations/instances_val2014_tongue_face.json'
cocoGt=COCO(annFile)

cats = cocoGt.loadCats(cocoGt.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = [cat['supercategory'] for cat in cats]
print('COCO supercategories:\n{}\n'.format(' '.join(nms)))

#initialize COCO detections api
resFile='/root/github/maskrcnn-benchmark/mymodels/maskrcnn_R50_FPN_1x/inference/coco_daosh_tongue_val/segm.json'
#resFile = resFile%(dataDir, prefix, dataType, annType)
cocoDt=cocoGt.loadRes(resFile)
imgIds=sorted(cocoGt.getImgIds())

print('imgIds: {}\n'.format(imgIds))
print('annType: {}\n'.format(annType))

imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]
# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
