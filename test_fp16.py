import time

import benchmark_utils

benchmark_utils.run_benchmark("fp16_models/detr-resnet50.onnx",
                              "test_image/001.jpg",
                              "FP16",
                              (512, 512))
time.sleep(30)
benchmark_utils.run_benchmark("fp16_models/resnet-v1-50.onnx",
                              "test_image/001.jpg",
                              "FP16",
                              (224, 224))
time.sleep(30)
benchmark_utils.run_benchmark("fp16_models/efficientnet-v2-b0.onnx",
                              "test_image/001.jpg",
                              "FP16",
                              (224, 224))
time.sleep(30)
benchmark_utils.run_benchmark("fp16_models/swin-tiny-patch4-window7-224.onnx",
                              "test_image/001.jpg",
                              "FP16",
                              (224, 224))
time.sleep(30)
benchmark_utils.run_benchmark("fp16_models/yolov5s.onnx",
                              "test_image/001.jpg",
                              "FP16",
                              (640, 640))
time.sleep(30)
benchmark_utils.run_benchmark("fp16_models/yolov5l.onnx",
                              "test_image/001.jpg",
                              "FP16",
                              (640, 640))
time.sleep(30)
benchmark_utils.run_benchmark("fp16_models/yolov8s.onnx",
                              "test_image/001.jpg",
                              "FP16",
                              (640, 640))
time.sleep(30)
benchmark_utils.run_benchmark("fp16_models/yolov8l.onnx",
                              "test_image/001.jpg",
                              "FP16",
                              (640, 640))