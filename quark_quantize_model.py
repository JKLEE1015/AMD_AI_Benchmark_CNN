import os  
import argparse
import onnx
import time  
import numpy as np  
from PIL import Image  
from pathlib import Path 
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationMethod  
from onnxruntime.quantization.quant_utils import QuantType  
from quark.onnx import ModelQuantizer  
from quark.onnx.quantization.config import Config, get_default_config  
#from utils import ImageDataReader, evaluate_onnx_model  

def preprocess_image(image_path):     
    image = Image.open(image_path)  
    image = image.resize((224, 224))  
    image_array = np.array(image).astype(np.float32)/255  
    image_array = np.transpose(image_array, (2, 0, 1))  
    input_data = np.expand_dims(image_array, axis=0)  
    return input_data  

def main(args):  
    # Setup the Input model  
    input_model_path = "fp32_models\\yolov8s.onnx" 
    output_model_path = "bf16_1_models\\yolov8s.onnx"  
    calibration_dataset_path = None
  

    # Get quantization configuration  
    quant_config = get_default_config("BF16")  
  
    # Defines the quantization configuration for the whole model  
    config = Config(global_quant_config=quant_config)
    quant_config.extra_options['UseRandomData'] = True  
    print("The configuration of the quantization is {}".format(config))  
  
    # Define the calibration data reader  
    #num_calib_data = 100  
    #calibration_dataset = ImageDataReader(calibration_dataset_path, input_model_path, data_size=num_calib_data, batch_size=32)  
  
    # Create an ONNX Quantizer  
    quantizer = ModelQuantizer(config)  
  
    # Quantize the ONNX model if the flag is set  
    if args.quantize:  
        quant_model = quantizer.quantize_model(model_input=input_model_path,   
                                               model_output=output_model_path,   
                                               calibration_data_reader=None)  
        print("Model Size:")  
        print("Float32 model size: {:.2f} MB".format(os.path.getsize(input_model_path)/(1024 * 1024)))  
        print("Int8 quantized model size: {:.2f} MB".format(os.path.getsize(output_model_path)/(1024 * 1024)))  
  
    # Evaluate the model if the flag is set  
    if args.evaluate:  
        print("Model Accuracy:")  
        top1_acc, top5_acc = evaluate_onnx_model(input_model_path, imagenet_data_path=calibration_dataset_path)  
        print("Float32 model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))  
        top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path=calibration_dataset_path)  
        print("Int8 quantized model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))  
        top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path=calibration_dataset_path, device='npu')  
        print("Int8 quantized model accuracy (NPU): Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))    
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Quantize and evaluate ONNX models.")  
    parser.add_argument('--model_input', type=str, default='models/resnet50.onnx', help='Path to the input ONNX model.')  
    parser.add_argument('--model_output', type=str, default='models/resnet50_quant.onnx', help='Path to save the quantized ONNX model.')  
    parser.add_argument('--calib_data', type=str, default='calib_data', help='Path to the calibration dataset.')  
    parser.add_argument('--quantize', action='store_true', help='Flag to quantize the model.')  
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model.') 
    parser.add_argument('--benchmark', action='store_true', help='Flag to benchmark the model.') 

    args = parser.parse_args()  
    main(args)  
