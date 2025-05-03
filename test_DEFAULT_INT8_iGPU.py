import time

import benchmark_utils

benchmark_utils.run_benchmark("DEFAULT_INT8_models/detr-resnet50_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (512, 512))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/resnet50_INT8_CNN_DEFAULT.onnx",
                             "test_image/001.jpg",
                             "FP32",
                             (224, 224))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/efficientnet-v2-b0_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (224, 224))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/swin-tiny-patch4-window7-224_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (224, 224))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/yolov5s_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (640, 640))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/yolov5l_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (640, 640))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/yolov8s_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (640, 640))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/yolov8l_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (640, 640))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/fastseg-large_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (2048, 1024))

time.sleep(30)

benchmark_utils.run_benchmark("DEFAULT_INT8_models/fastseg-small_INT8_CNN_DEFAULT.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (2048, 1024))
