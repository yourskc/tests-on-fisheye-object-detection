import onnx

model_path = "yolo_object_detection.onnx"
model = onnx.load(model_path)
graph_inputs = model.graph.input    
for model_input in graph_inputs:
	input_name = model_input.name
	input_shape = []
	for dim in model_input.type.tensor_type.shape.dim:
		if dim.HasField("dim_value"):
			input_shape.append(dim.dim_value)
		else:
			# This dimension is dynamic (e.g., batch size), often represented by -1
			input_shape.append(-1)
	print(f"Input Name: {input_name}, Input Shape: {input_shape}")    
