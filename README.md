# DeeSIL
new code version of DeeSIL

## set up your env 
using conda, install the required packages (of `package-list.txt`)

```conda create -n deesil --file package-list.txt```

and add `utils/` to your env path
```conda develop path_to_utils/```


## train the first state
edit `configs/scratch_ilsvrc_s10.cf` to set your storing paths (e.g. `path_to_model_dir`)
then train the first state model using:

```python code/scratch.py configs/scratch_ilsvrc_s10.cf```

## extract all the features
edit `configs/features_extraction_ilsvrc_s10.cf` to set your storing paths (e.g. `path_to_model_dir` and the `root_path` (that will be used then to store the `features/`, `svms/` and `predictions/` folder))
then extract the required features using:

```python code/features_extraction.py configs/features_extraction_ilsvrc_s10.cf```

## train the DeeSIL classifiers
run the following commands, it will train the classifiers of DeeSIL for each state and store them in the right folder (replace `root_path` accordingly to the previous step)

```python codes/train_deesil.py 1 0.0001 0 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 1 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 2 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 3 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 4 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 5 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 6 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 7 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 8 100 ilsvrc s10 root_path
python codes/train_deesil.py 1 0.0001 9 100 ilsvrc s10 root_path
```

(1 and 0.0001 stands for the C and tol parameters and are already optimized)

## compute the predictions files 
run the following commands, it will compute the predictions files of DeeSIL store them in the right folder (replace `root_path` accordingly to the previous steps)

```python codes/compute_predictions_deesil.py ilsvrc s10 root_path```

## evaluate DeeSIL 
run the following commands, it will evaluate DeeSIL accordingly to the predictions files of DeeSIL computed at the previous step (replace `root_path` accordingly to the previous steps)

```python codes/eval_deesil.py ilsvrc s10 root_path```

### Results:
You should get for ILSVRC s10:
```
batch 1, top1 = 0.840, top5 = 0.949
batch 2, top1 = 0.654, top5 = 0.885
batch 3, top1 = 0.564, top5 = 0.844
batch 4, top1 = 0.501, top5 = 0.793
batch 5, top1 = 0.457, top5 = 0.755
batch 6, top1 = 0.427, top5 = 0.728
batch 7, top1 = 0.405, top5 = 0.705
batch 8, top1 = 0.380, top5 = 0.682
batch 9, top1 = 0.366, top5 = 0.666
batch 10, top1 = 0.351, top5 = 0.648
=================================================
===================  TOTAL  =====================
top1 = 0.456, top5 = 0.745
```