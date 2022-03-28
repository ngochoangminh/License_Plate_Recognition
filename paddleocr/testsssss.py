import sys
sys.path.append("/home/ngoc/work/ai_acd/PaddleOCR")

import cv2
import time
import numpy as np
import onnxruntime as rt
from CarDetect.onnx_infer import detect_car
from LPdetect.onnx_infer import LP_detect
from loguru import logger
from tools.infer.predict_det import TextDetector
from tools.infer.predict_rec import TextRecognizer
import tools.infer.utility as utility

"""
def draw_text_det_res(dt_boxes, src_im,text='unknown'):
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    # if text is not None:
    #     cv2.putText(src_im, text, box[1],fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 50), thickness=1, lineType=2)
    return src_im

def visualize(img_raw, h_min, w_min, h_max, w_max, text):
    cv2.rectangle(img_raw, (w_min, h_min), (w_max, h_max), (255,255,0), 1)
    cv2.putText(img_raw, text, (w_min, h_min-2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5,
                color=(255, 255, 50), 
                thickness=1)
    cv2.imshow('a',cv2.cvtColor(img_raw,cv2.COLOR_RGB2BGR))
    return img_raw
"""

@logger.catch
def single_image(yolox_sess, wpod_sess, text_recognizer, img_raw):
    res = []
    r = dict()
    car_dets = detect_car(yolox_sess, img_raw)
    #print(car_dets)
    for i in range(len(car_dets)):
        lps = []
        add=[]
        for j in range(4):
            lps.append(int(car_dets[i][j]))
        add.append(lps)
        print('obj add:',add)
        car_det = img_raw[lps[1]:lps[3],lps[0]:lps[2],:]
        tlp, lp_type = LP_detect(wpod_sess,car_det)
        # cv2.imshow("Bien so", cv2.cvtColor(tlp[0], cv2.COLOR_RGB2BGR))
        # cv2.waitKey()

        logger.info('lp_type')
        if len(tlp)>0:
            print(len(tlp))
            tl = cv2.cvtColor(tlp[0], cv2.COLOR_RGB2BGR)
            print(type(tl), tl.shape)
            # cv2.imshow("LP", tl)
            # cv2.waitKey()
            if lp_type[0]==1:
                logger.info('Lp type is 1, lp shape =  '+str(tl.shape))
                cv2.imshow("type 1, after recog", tl)
                cv2.waitKey()
                rec_res, _ = text_recognizer([tl])
                # rec_res = rec_res[0][0]
                print("rec_res: ",rec_res[0])
            else:
                logger.info('LP is type 2, lp shape = '+str(tl.shape))
                # cv2.imshow("line 1", tl[:100,:,:])
                # cv2.waitKey()
                # cv2.imshow("line 2", tl[100:,:,:])
                # cv2.waitKey()
                rec_res_1, _ = text_recognizer([tl[:100,:,:]])
                rec_res_2, _ = text_recognizer([tl[:100,:,:]])
                rec_res = rec_res_1[0][0]+rec_res_2[0][0]
                print(rec_res_1,rec_res_2, rec_res)
   
    """if len(dt_boxes)>0:
            # bien ngan: 165x330, bien dai: 110x520
            a = np.array(dt_boxes[0]).astype(np.int32).reshape(-1, 2)
            w_min = np.min(a[:,0])+lps[0]
            h_min = np.min(a[:,1])+lps[1]
            w_max = np.max(a[:,0])+lps[0]
            h_max = np.max(a[:,1])+lps[1]
            # print(lps)
            # print(h_min,w_min,'\t',h_max,w_max)
            scl = round((w_max-w_min)/(h_max-h_min), 2)
            logger.info('ti le bien: '+str(scl))
            cv2.rectangle(img_raw, (w_min, h_min-50), (w_max, h_max), (255,255,0), 1)
            cv2.imshow('bien',img_raw)
            cv2.waitKey()
            if scl > 3:
                lp = img_raw[h_min:h_max, w_min:w_max,:]            
                rec_res, _ = text_recognizer([lp])
                rec_res = rec_res[0][0]
            else:
                h_avg = round((h_max-h_min)/2)
                logger.info('h avg = '+str(h_avg))
                line_1 = img_raw[h_min:h_min+h_avg, w_min:w_max,:]
                line_2 = img_raw[h_min+h_avg:h_max, w_min:w_max,:]
                cv2.imshow('a',line_1)
                cv2.waitKey()
                cv2.imshow('b',line_2)
                cv2.waitKey()
                rec_res_1, _ = text_recognizer([line_1])
                rec_res_2, _ = text_recognizer([line_2])
                rec_res = rec_res_1[0][0]+rec_res_2[0][0]
            if rec_res is not None:
                res.append(rec_res)
                r['lp']=rec_res
            else:
                res.append('Unknown')
                r['lp']='Unknown'

            # cv2.rectangle(img_raw, (w_min, h_min), (w_max, h_max), (255,255,0), 1)
            # cv2.putText(img_raw, rec_res[0][0], (w_min, h_min-2),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            #             fontScale=0.5,
            #             color=(255, 255, 50), 
            #             thickness=1)
            # cv2.imshow('a',cv2.cvtColor(img_raw,cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    r['car_dets']=add
    # print('add',add)
    print('r\n',r)"""
    
    return res
if __name__=="__main__":
    vid_path = '/home/ngoc/work/ai_acd/data/kv_san_2.mp4'
    car1 = "./images/test2.jpg"
    car2 = './images/test3.jpg'
    car4 = './images/test5.jpg'
    yolox_onnx = './license_plate/CarDetect/cuong.onnx'
    wpod_onnx = './license_plate/LPdetect/wpod.onnx'
    count = 1
    warmup = True

    args = utility.parse_args()
    print("args: ", args.rec_algorithm, args.det_algorithm)
    
    text_detector = TextDetector(args)
    text_recognizer = TextRecognizer(args)
    print(text_recognizer)
    wpod_sess = rt.InferenceSession(wpod_onnx, providers=['CUDAExecutionProvider'])
    yolox_sess = rt.InferenceSession(yolox_onnx, providers=['CUDAExecutionProvider'])
    # warmup
    if warmup:
        img = np.random.uniform(0, 255, [32, 320, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.rec_batch_num))

    img_raw = cv2.imread(car1)
    ts = time.time()
    res = single_image(yolox_sess, wpod_sess, text_recognizer, img_raw)

    # print('###### FPS: {:.2f} ----- LP: {:s}'.format(round(1/(time.time()-ts),2),res[0][0][0]))

"""
    # Infer Videos
    cap = cv2.VideoCapture(vid_path)
    while (cap.isOpened()):
        res, frame = cap.read()
        try: 
            ts = time.time()
            result = single_image(yolox_sess, text_detector, text_recognizer, frame)
            # print('###### FPS: {:.2f} ----- LP: {:s}'.format(round(1/(time.time()-ts),2), result[0][0][0]))
            print('FPS: ',round(1/(time.time()-ts),2),'\n \t List LP: ',result)
        except:
            pass
    cv2.destroyAllWindows()
"""
