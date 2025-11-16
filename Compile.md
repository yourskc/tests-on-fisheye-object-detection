Reference :

drp-ai_tvm

[https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/v2.3.0/tutorials/tutorial_RZV2H.md](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/v2.3.0/tutorials/tutorial_RZV2H.md)

DRP-AI Translator i8 [V1.10]

[https://www.renesas.com/en/software-tool/drp-ai-translator-i8?srsltid=AfmBOoq8D0ml6J1RcvckTFiM3mUt9L5Y-fP0LhTlb_kEZrL66_9Yz1xh#downloads](https://www.renesas.com/en/software-tool/drp-ai-translator-i8?srsltid=AfmBOoq8D0ml6J1RcvckTFiM3mUt9L5Y-fP0LhTlb_kEZrL66_9Yz1xh#downloads)

DRP-AI Quantizer

[https://www.renesas.com/en/document/mas/drp-ai-quantizer-v130-users-manual?r=25472906](https://www.renesas.com/en/document/mas/drp-ai-quantizer-v130-users-manual?r=25472906)

drp-ai_driver

[https://github.com/renesas-rz/rzv2h_drp-ai_driver](https://github.com/renesas-rz/rzv2h_drp-ai_driver)



Compare the compile parameter and run result on V2H 

## 1. Case 1 

Compile command

```
python3 compile_onnx_model_quant.py \
    ./yolo_object_detection.onnx \
    -o yolo3 \
    -t $SDK \
    -d $TRANSLATOR \
    -c $QUANTIZER 
```
結果   
Default input shape : 1,3,244,244
會有錯誤訊息
too many indices for array
Message :
```
Input_node_name: in
         Address: 0x0
         Channel: 3
         Width  : 640
         Height : 480

Output_node_name: out
         Address: 0x198c00
         Channel: 3
         Width  : 224
         Height : 224
```

這一版送到 V2H 後執行 R01_object_detection_fisheye/object_detection

Error : Segmentation Fault


## 2. Case 2 

Compile command

```
python3 compile_onnx_model_quant.py \
    ./yolo_object_detection.onnx \
    -o yolo3 \
    -t $SDK \
    -d $TRANSLATOR \
    -c $QUANTIZER \
    -v 100 \
    -s 1,3,640,640
```    
    
一開始會先列出 input shape 640x640    

會有錯誤訊息
```
too many indices for array  -> Check 
```

結果 
```
Input_node_name: in
         Address: 0x0
         Channel: 3
         Width  : 640
         Height : 480

Output_node_name: out
         Address: 0x6bd000
         Channel: 3
         Width  : 640
         Height : 640
```

在 V2H 執行 object_detection, 使用
S-YUE Fisheye Camera

```
Total time: about 780 ms ( Pre+Inf+Post)
Bicycle 100%
Person  100%
```





    
    
