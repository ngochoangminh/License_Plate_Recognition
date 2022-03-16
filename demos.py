import torch
import onnxruntime as rt
import time
import cv2
from PIL import Image
import numpy as np
from torch.autograd import Variable
from LPrecog.infer import resizeNormalize, strLabelConverter, crnn_pred
from LPdetect.onnx_infer import LP_detect

def imgprocess(img):
    img = cv2.resize(img,(100,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#/255.
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    img.sub_(0.5).div_(0.5)
    return img

def single_image_infer(sess_wpod, sess_crnn, img_path, converter):
    res = []
    tlp ,lp_type = LP_detect(sess_wpod, img_path)
    for i in range(len(tlp)):
        if lp_type==1:
            tl = cv2.cvtColor(tlp[i], cv2.COLOR_RGB2BGR)
            _, sim_pred = crnn_pred(sess_crnn, imgprocess(tl), converter)
            res.append(sim_pred)
        else:
            h = round(tlp[i].shape[0]/2)
            tl = cv2.cvtColor(tlp[i], cv2.COLOR_RGB2BGR)
            _, r1 = crnn_pred(sess_crnn, imgprocess(tl[:h,:,:]), converter)
            _, r2 = crnn_pred(sess_crnn, imgprocess(tl[h:,:,:]), converter)
            res.append((r1+r2))
    return res

if __name__ == "__main__":
    img_path = "./images/xemay.jpg"
    wpod_net_path = "./LPdetect/wpod.onnx"
    crnn_model_path = "./LPrecog/crnn.onnx"
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ncl = len(alphabet)+1 #39

    converter = strLabelConverter(alphabet)
    transformer = resizeNormalize((100, 32))
    sess_crnn = rt.InferenceSession(crnn_model_path, providers=['CUDAExecutionProvider'])
    sess_wpod = rt.InferenceSession(wpod_net_path, providers=['CUDAExecutionProvider'])

    t0=time.time()
    sim_pred = single_image_infer(sess_wpod, sess_crnn, img_path, converter)
    print('Time infer: ',time.time()-t0, ' Prediction: %-20s' % (sim_pred))
    