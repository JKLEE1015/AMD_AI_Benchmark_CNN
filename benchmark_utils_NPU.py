import time

from PIL import Image
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path


def run_benchmark(model_path, image_path, quantization, input_size):
    model = onnx.load(model_path)

    providers = ['VitisAIExecutionProvider']
    cache_dir = Path(__file__).parent.resolve()
    provider_options = [{
                'cacheDir': str(cache_dir),
                'cacheKey': 'modelcachekey',
                'xcbin':'C:\Program Files\RyzenAI\1.4.0\voe-4.0-win_amd64/xcbin',
              'ai_analyzer_visualization': True,
              'ai_analyzer_profiling': True,    
    }]

    session = ort.InferenceSession(model.SerializeToString(), providers=providers, provider_options=provider_options)

    data_format = np.float32 if quantization == "FP32" else np.float16

    image = Image.open(image_path)
    # Reshape the array to match the input shape expected by the model
    image = image.resize(input_size)
    image_array = np.array(image)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.ascontiguousarray(image_array)
    # print(image_array.flags['C_CONTIGUOUS'])
    # Add a batch dimension to the input image
    input_tensor = np.expand_dims(image_array, axis=0)
    input_data = input_tensor.astype(data_format)

    latencies = []

    for _ in range(100):
        session.run(None, {session.get_inputs()[0].name: input_data})

    for _ in range(1000):
        start_time = time.perf_counter()
        session.run(None, {session.get_inputs()[0].name: input_data})
        latency = (time.perf_counter() - start_time) * 1000
        latencies.append(latency)

    print(f"Model Name " + model_path)
    print(f"Average Latency: {np.mean(latencies):.2f} ms")
    print("==========================================")


