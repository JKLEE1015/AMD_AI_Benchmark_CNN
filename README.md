# AMD_AI_Benchmark_CNN

The Ryzen AI compiler NPU supports input models quantized to either INT8 or BF16 format:

CNN models: INT8 or BF16
Transformer models: BF16

AMD’s latest NPU and GPU devices natively support BF16

model convert

FP32 -> XINT8
python -m quark.onnx.tools.random_quantize --input_model fp32_models\detr-resnet50.onnx --quant_model XINT8_models\detr-resnet50.onnx --config XINT8


FP32 -> BF16
python -m quark.onnx.tools.convert_fp32_to_bf16 --input fp32_models\detr-resnet50.onnx --output bf16_models\detr-resnet50.onnx --format with_cast

FP16 ->BF16
python -m quark.onnx.tools.convert_fp16_to_bf16 --input  fp16_models\detr-resnet50.onnx --output bf16_models\detr-resnet50.onnx --format with_cast

