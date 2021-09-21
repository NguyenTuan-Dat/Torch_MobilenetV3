import torch
from MobilenetV3 import mobilenetv3_small_multitask

def load_state_dict(model, state_dict):
    all_keys = {k for k in state_dict.keys()}
    for k in all_keys:
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    if len(pretrained_dict) == len(model_dict):
        print("all params loaded")
    else:
        not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
        print("not loaded keys:", not_loaded_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

PATH_TO_MODEL = "/Users/ntdat/Downloads/112_Classify_Epoch_20_Batch_17320_93.108_95.396_Time_1631854322.1392875_checkpoint.pth"

net = mobilenetv3_small_multitask()
load_state_dict(net, torch.load(PATH_TO_MODEL, map_location="cpu"))
net.eval()
dummy_input = torch.randn(1, 3, 112, 112)
torch.onnx.export(net, dummy_input,
                  "/Users/ntdat/Downloads/20210917_classify_112_20.onnx",
                  # output_names=["0_glasses","1_mask","2_hat"])
                output_names=["0_glasses","1_mask"])