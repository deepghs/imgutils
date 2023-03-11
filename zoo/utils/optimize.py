import onnxoptimizer
import onnxsim


def onnx_optimize(model):
    model = onnxoptimizer.optimize(model)
    model, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model
