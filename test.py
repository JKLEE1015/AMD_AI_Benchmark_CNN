import onnx

onnx.checker.check_model("fp16_models/detr-resnet50.onnx")