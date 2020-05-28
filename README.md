# U-IMG2DSM: Unpaired Simulation of Digital Surface Models with Generative Adversarial Networks
The Code for "U-IMG2DSM: Unpaired Simulation of Digital Surface Models with Generative Adversarial Networks". []
```
M. E. Paoletti, J. M. Haut, P. Ghamisi, N. Yokoya, J. Plaza and A. Plaza.
U-IMG2DSM: Unpaired Simulation of Digital Surface Models with Generative Adversarial Networks.
IEEE Geoscience and Remote Sensing Letters.
DOI: 10.1109/LGRS.2020.2997295.
Accepted for publication, June 2020.
```

![UIMG2DSM](https://github.com/mhaut/Uimg2dsm/blob/master/images/generated.png)

### Run code

```
cd checkpoints/
python join_checkpoint.py
cd ../results/classifications_results/generated_maps/
python join_dset.py


# To generated DSM from IMG
python test_batch.py --trainer UNIT --config ./configs/unit_img2dsm_folder.yaml --a2b 1 --input ./dataset/testA/ --output_folder ./results/outputs --checkpoint ./outputs/unit_img2dsm_folder/checkpoints/gen_00700000.pt
# To train new network
python train.py --trainer UNIT --config ./configs/unit_img2dsm_folder.yaml


# To get metrics
cd results
open Octave in ./results
load pkg image
launch calculateRMSEZNCC
OR
cd results/classifications_results
python classify.py <Algorithm>
python classify.py RF

```

Reference code: https://github.com/mingyuliutw/UNIT
