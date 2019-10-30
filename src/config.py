import os

#root_path = "C:/Users/appleternity/appleternity/workspace/EmoContext"
root_path = "/home/appleternity/workspace/2019Fall/deep_learning/EmoContext"
data_path = os.path.join(root_path, "data")
model_path = os.path.join(root_path, "model")
result_path = os.path.join(root_path, "result")

label_mapping = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "others": 3
}
label_mapping_reverse = {val:key for key, val in label_mapping.items()}
