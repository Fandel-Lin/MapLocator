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
