from tkinter import Frame
import cv2
import time
import numpy as np
import onnxruntime as rt
from LPrecog.infer import resizeNormalize, strLabelConverter
from demos import single_image_infer, single_image

def video_infer(sess_yolox, sess_wpod, sess_crnn, vid_path, converter):
    cap = cv2.VideoCapture(vid_path)
    while (cap.isOpened()):
        res, frame = cap.read()
        try:
            t0=time.time()
            # res = single_image_infer(sess_wpod, sess_crnn, frame, converter)
            res = single_image(sess_yolox, sess_wpod, sess_crnn, frame, converter)
            print('FPS: ', round(1/(time.time()-t0), 2), ' ----- Result: ',res)
        except:
            pass
    cv2.destroyAllWindows()


    
if __name__ == "__main__":
    vid_path = "/home/ngoc/work/ai_acd/data/kv_san_2.mp4"
    yolox_nano_path = "./CarDetect/yolox_nano.onnx"
    wpod_net_path = "./LPdetect/wpod.onnx"
    crnn_model_path = "./LPrecog/crnn.onnx"
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ncl = len(alphabet)+1 #39

    converter = strLabelConverter(alphabet)
    # transformer = resizeNormalize((100, 32))
    sess_yolox = rt.InferenceSession(yolox_nano_path, providers=['CUDAExecutionProvider'])
    sess_crnn = rt.InferenceSession(crnn_model_path, providers=['CUDAExecutionProvider'])
    sess_wpod = rt.InferenceSession(wpod_net_path, providers=['CUDAExecutionProvider'])

    
    # sim_pred = single_image_infer(sess_wpod, sess_crnn, img_path, converter)
    # print('Time infer: ',time.time()-t0, ' Prediction: %-20s' % (sim_pred))
    video_infer(sess_yolox, sess_wpod, sess_crnn, vid_path, converter)
    
