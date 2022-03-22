# License_Plate_Recognition

```
pip install -r requirement.txt
```


## Convert PaddleOCR to ONNX
Download weight and extract

```
wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/chinese/ch_PP-OCRv2_rec_infer.tar && tar -xf ch_PP-OCRv2_rec_infer.tar

Struct_paddle_weights\ 
├── inference.pdmodel  
├── inference.pdiparams.info  
└── inference.pdiparams
```
Convert:
```
pip install git+https://github.com/PaddlePaddle/Paddle2ONNX.git

paddle2onnx --model_dir path_to_paddle_weights \
--save_file path_to_dest_onnx \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--opset_version 11 \
--input_shape_dict="{'x': [1, 3, 32, 128]}"
```
