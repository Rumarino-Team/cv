import requests
import hashlib
from collections import namedtuple
import glob
ModelWeights = namedtuple("ModelWeights",["name", "url", "path", "hash"])
model_paths = {

        "dav2_s": ModelWeights('dav2_s', "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",\
                "weights/dav2_s.pt", "00123124"),
        "dav2_b": ModelWeights('dav2_b',"https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
                               "weights/dav2_b.pt", "0124887124")
        }



def helper_calculate_hash(filename, algorithm="sha256", chunk_size=8192)-> str:
    h = hashlib.new(algorithm)  # e.g. "md5", "sha1", "sha256"
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

    

def choose_download_model():
    print("Choose a model to download:")
    picked_model : int = -1 
    while picked_model < 0 or picked_model > len(model_paths):
        for idx, model_name in enumerate(model_paths):
            print(f"({idx-1}) : {model_name}")
        picked_model = input(f"Choose a model between <{0}, {len(model_paths) -1}: ")

    model_weights = list(model_paths)[picked_model]
    if not glob.glob(model_weights.path):
        r = requests.get(model_weights.url, stream=True)
        with open(model_weights.path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    if model_weights.hash != helper_calculate_hash(model_weights.path):
        raise Exception("The Model in disk doesnt have the same hash. The model its probably corrupted.")

    print(" Model has been Downloaded correctly!!")



if __name__ == "__main__":
    print("Options:")
    print("1) Calculate hash of models weights.")
    print("2) Download the models weights.")
    option = int(input("Option: "))

    if option == 1:
        choose_download_model()
    elif option ==2:
        raise Exception("Not yet Implemented")

    else:
        print("Try Again  Stupid.")
