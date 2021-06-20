released code for NP-DRAW paper

## Dependencies
```bash
# the following command will install torch 1.6.0 and other required packages 
conda env create -f environment.yml # edit the last link in the yml file for the directory
conda activate npdraw 
```
## Pretrained Model 
Pretrained model will be available [here] 
To use the pretrained models, download the `exp` folder and put it under the project root directory.

#### Testing the pretrained NPDRAW model:
The following commands test the FID score of the NPDRAW model. 
The commands output the CD and EMD on the test/validation sets.

```bash
# Usage:
bash scripts/local_sample.sh exp/stoch_mnist/cat_vloc_at/0208/p5s5n36vitBinkl1r1E3_K50w5sc0_gs_difflr_b500/ckpt_epo799.pth 

python test.py configs/recon/airplane/airplane_recon_add.yaml \
    --pretrained pretrained/recon/airplane_recon_add.pt
python test.py configs/recon/car/car_recon_add.yaml \
    --pretrained pretrained/recon/car_recon_add.pt
python test.py configs/recon/chair/chair_recon_add.yaml \
    --pretrained pretrained/recon/chair_recon_add.pt
```
The pretrained model's auto-encoding performance is as follows:
| Dataset  | Metrics  | Ours  | Oracle |
|----------|----------|-------|--------|
| Airplane | CD x1e4  | 0.966 |  0.837 |
|          | EMD x1e2 | 2.632 |  2.062 |
| Chair    | CD x1e4  | 5.660 |  3.201 |
|          | EMD x1e2 | 4.976 |  3.297 |
| Car      | CD x1e4  | 5.306 |  3.904 |
|          | EMD x1e2 | 4.380 |  3.251 |

