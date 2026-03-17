## How to run 

Please run '0_map_segmentation.py' with the following parameters: 

```
--input: path to input image
--output_dir: path to output directory
--intermediate_dir: path to intermediate output directory
--device: default to "cuda"
--controller: default to "auto" (choose among "topo", "pp1300", "nickel", "auto")
```

Please adjust the controller based on the datasets, or use "auto" to proceed with a classifier automatically. 

## Setup

The requirements are listed in the 'requirements.txt' file.

For fast_slic, please refer to 'https://github.com/Algy/fast-slic' regarding performance under avx2 support. 

## Environment 

The script is tested under the following three environments:

(1) Python 3.12.8, Windows Version 10.0.26200, NVIDIA GeForce RTX 4060 (8 GB), Intel i9-13900H at 2.60 GHz, 48 GB RAM at 4800 MT/s.
(2) Python 3.12.3, Windows Version 10.0.26200, NVIDIA GeForce RTX 3090 (24 GB), AMD Ryzen 9 5900X at 3.70 GHz, 128 GB RAM at 3200 MT/s.
(3) Python 3.12.9, Windows Version 10.0.22631, NVIDIA RTX A6000 (48 GB), Intel Xeon w9-3595X at 1.99 GHz, 512 GB RAM at 4800 MT/s.
