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

Next, prepare a `abaw.csv` file in the root directory. The header of the file is [img_name,label]. Here, "img_name" means the frame id in the competition dataset, "label" means the fake label given by the model.
# CNNs train
To train our model, run:

`python vote.py`

# predict
The given `abaw.csv` is result given by Claude in our experiment.

The model parameters should be placed in "./model/model{*}.py", for example, for model 0, it should be placed in  "./model/model0.py"

To  test and write the result into a txt file, you should prepare "CVPR_6th_ABAW_CE_test_set_sample.txt" in the root directory, then run:

`python predict.py`

