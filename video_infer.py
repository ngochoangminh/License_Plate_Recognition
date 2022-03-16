from tkinter import Frame
import cv2
import time
import numpy as np
import onnxruntime as rt
from LPrecog.infer import resizeNormalize, strLabelConverter
from demos import single_image_infer

def video_infer(sess_wpod, sess_crnn, vid_path, converter):
    cap = cv2.VideoCapture(vid_path)
    while (cap.isOpened()):
        res, frame = cap.read()
        while res==True:
            res = single_image_infer(sess_wpod, sess_crnn, frame, converter)
            print(res)

    cv2.destroyAllWindows()


    
if __name__ == "__main__":
    img_path = "./images/IMG_1425.jpg"
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
    
