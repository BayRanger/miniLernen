from selfAttention import *
import onnxruntime as rt
import onnx
import numpy as np


def onnxruntime_check(onnx_path, input_dicts, torch_outputs):
    onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    sess = rt.InferenceSession(onnx_path)
    # outputs = self.get_output_names()
    # latent input
    # data = np.zeros((4, 77), dtype=np.int32)
    result = sess.run(None, input_dicts)

    for i in range(0, len(torch_outputs)):
        ret = np.allclose(result[i], torch_outputs[i].detach().numpy(), rtol=1e-03, atol=1e-04, equal_nan=False)
        if ret is False:
            res = result[i] - torch_outputs[i].detach().numpy()
            print(f"output[{i}] has maximum diff {np.max(res)}")
            print("Error onnxruntime_check, break!")
            return
    print(f"======================= {onnx_path} verify done!")



if __name__ =="__main__":
    #load model and parameters
    batch_size = 1
    seq_length = 12
    input_dim = 16
    model_name = "self_attention"
    self_atten  = SelfAttention(input_dim)
    #create dummpy input
    self_atten.load_state_dict(torch.load('atten_weights.pth'))

    input_tensor = torch.ones(1, seq_length, input_dim)

    onnx_path = f'{model_name}.onnx'
    #export the onnx model
    torch.onnx.export(self_atten,
                      input_tensor,
                      onnx_path,
                      verbose = False,
                      input_names= ["query"],
                      output_names= ["attention_weights"] ,
                      opset_version=14
                      )

    print(f"======================={onnx_path} export onnx done!")

    #check the result
    torch_result = self_atten(input_tensor)
    #print(torch_result)

    input_dicts = {}
    input_dicts["query"] = input_tensor.numpy()
    onnxruntime_check(onnx_path, input_dicts, [torch_result])
    print("success exit")