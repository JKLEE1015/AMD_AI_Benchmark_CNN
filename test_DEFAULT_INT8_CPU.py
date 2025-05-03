import time

import benchmark_utils_CPU

benchmark_utils_CPU.run_benchmark("int8_models/detr-resnet50.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (512, 512))

time.sleep(30)

benchmark_utils_CPU.run_benchmark("int8_models/resnet-v1-50.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (224, 224))

time.sleep(30)

benchmark_utils_CPU.run_benchmark("int8_models/efficientnet-v2-b0.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (224, 224))

time.sleep(30)

benchmark_utils_CPU.run_benchmark("int8_models/swin-tiny-patch4-window7-224.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (224, 224))

time.sleep(30)

benchmark_utils_CPU.run_benchmark("int8_models/yolov5s.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (640, 640))

time.sleep(30)

benchmark_utils_CPU.run_benchmark("int8_models/yolov5l.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (640, 640))

time.sleep(30)

benchmark_utils_CPU.run_benchmark("int8_models/yolov8s.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (640, 640))

time.sleep(30)

benchmark_utils_CPU.run_benchmark("int8_models/yolov8l.onnx",
                              "test_image/001.jpg",
                              "FP32",
                              (640, 640))


