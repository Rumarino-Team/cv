import hashlib
import subprocess
from collections import namedtuple
import glob
ModelWeights = namedtuple("ModelWeights",["name", "url", "path", "hash"])
model_paths = {

        "dav2_s": ModelWeights('dav2_s', "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",\
                "weights/dav2_s.pt", "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"),
        "dav2_b": ModelWeights('dav2_b',"https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
                               "weights/dav2_b.pt", "0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328")
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
            print(f"({idx}) : {model_name}")
        picked_model = int(input(f"Choose a model between [{0}, {len(model_paths) -1}]: "))

    model_weights = list(model_paths.items())[picked_model][1]
    if not glob.glob(model_weights.path):
        try:
            subprocess.run(
                    ["curl", "-L", model_weights.url, "-o", model_weights.path],
                    check=True)
            print(f"Downloaded: {model_weights.path}")

        except subprocess.CalledProcessError as e:
            print("Download failed:", e)
    if model_weights.hash != helper_calculate_hash(model_weights.path):
        raise Exception("The Model in disk doesnt have the same hash. The model its probably corrupted.")




if __name__ == "__main__":
    choose_download_model()

