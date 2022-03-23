import torch
import onnxruntime as rt
import time
import cv2
# from PIL import Image
import numpy as np
from CarDetect.onnx_infer import detect_car
from LPrecog.infer import strLabelConverter, crnn_pred
from LPdetect.onnx_infer import LP_detect, img_process

def imgprocess(img):
    img = cv2.resize(img,(100,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#/255.
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    img.sub_(0.5).div_(0.5)
    return img


def single_image_infer(sess_wpod, sess_crnn, img_raw, converter):
    res = []
    tlp ,lp_type = LP_detect(sess_wpod, img_raw)
    if len(tlp)>0:
        for i in range(len(tlp)):
            if lp_type[i]==1:
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

def single_image(sess_yolox, sess_wpod, sess_crnn, img_raw, converter):
    res = []
    dets = detect_car(sess_yolox, img_raw)
    for i in range(len(dets)):
        
        if int(dets[i][5]) == 2:
            lp = []
            for j in range(4):
                # print(int(dets[i][j]))
                lp.append(int(dets[i][j]))
            # cv2.imshow("Bien so", cv2.cvtColor(img_raw[lp[1]:lp[3],lp[0]:lp[2],:], cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            tlp ,lp_type = LP_detect(sess_wpod, img_raw[lp[1]:lp[3],lp[0]:lp[2],:])
            if len(tlp)>0:
                for i in range(len(tlp)):
                    if lp_type[i]==1:
                        tl = cv2.cvtColor(tlp[i], cv2.COLOR_RGB2BGR)
                        _, sim_pred = crnn_pred(sess_crnn, imgprocess(tl), converter)
                        lp.append(sim_pred)
                    else:
                        tl = cv2.cvtColor(tlp[i], cv2.COLOR_RGB2BGR)
                        _, r1 = crnn_pred(sess_crnn, imgprocess(tl[:100,:,:]), converter)
                        _, r2 = crnn_pred(sess_crnn, imgprocess(tl[100:,:,:]), converter)
                        lp.append((r1+r2))
            # print(lp)
            if lp is not None:
                res.append(lp)
    return res

if __name__ == "__main__":
    img_path = "./images/luxa.jpg"
    wpod_net_path = "./LPdetect/wpod.onnx"
    crnn_model_path = "./LPrecog/crnn.onnx"
    yolox_nano_path = "./CarDetect/yolox_nano.onnx"
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ncl = len(alphabet)+1 #39
    img_raw = cv2.imread(img_path)
    
    converter = strLabelConverter(alphabet)
    sess_crnn = rt.InferenceSession(crnn_model_path, providers=['CUDAExecutionProvider'])
    sess_wpod = rt.InferenceSession(wpod_net_path, providers=['CUDAExecutionProvider'])
    sess_yolox = rt.InferenceSession(yolox_nano_path, providers=['CUDAExecutionProvider'])
    
    t0=time.time()
    sim_pred = single_image(sess_yolox, sess_wpod, sess_crnn, img_raw, converter)
    print('Time infer: ',time.time()-t0, ' Prediction: %-20s' % (sim_pred))
    