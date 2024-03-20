# abaw-ce-ustc
This github repository holds the code for the USTC-AC team to participate in the 6th ABAW competition.

# start
`pip install -r requirements.txt`

# Data process
```
video_root="./abaw-test/videos/"
image_root="./abaw-test/images_raw/"
save_root="./abaw-test/images_aligned/"
```
The first step is to change video_root, image_root, save_root in `Z-CER/data/preprocessing.py` to the path of the corresponding file.

Then, run `python preprocessing.py`.

Next, prepare a `train.csv` and a `test.csv` file in the root directory. The header of both files is [img_name,label], which corresponds to the full path and label of the image file in the RAF-DB dataset.

# Visual language Model
Run `python Visual_language_Model.py` several times until the error length becomes 0.

Then, run `python Label_processing.py`

# CNNs train
`python vote.py`

# predict
`python predict.py`

