"""

git clone https://github.com/meijieru/crnn.pytorch.git

cd crnn.pytorch

python cvt2onnx.py

"""


import torch
import models.crnn as crnn

model_path = './LPrecog/crnn.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
ncl = len(alphabet)+1 #39

model = crnn.CRNN(32, 1, ncl, 256)
if torch.cuda.is_available():
    model = model.cpu()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
model.eval()

X = torch.ones(1, 1, 32, 100).to(torch.device("cpu"))

torch.onnx.export(model,                     # model being run
                  X,                         # model input (or a tuple for multiple inputs)
                  "crnn.onnx",               # where to save the model (can be a file or file-like object)
                  opset_version=10,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}}
                                )