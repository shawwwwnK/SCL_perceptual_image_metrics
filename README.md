# Perceptual Image Metrics

## Requirements
- Create conda environment and install required packages:
    ```
    conda create --name myenv python=3.8.2
    conda activate myenv
    python -m pip install -r requirements.txt
    ```
- Add path to the repo to `PYTHONPATH`:
    ```
    export PYTHONPATH=$PYTHONPATH:<path_to_repo>
    ``` 

- Other requirements

    Pytorch and the LPIPS package is required to run the metrics
    
    To install pytorch:
    ```
    conda install pytorch
    ```
    
    To install the LPIPS package, follow https://github.com/richzhang/PerceptualSimilarity
    
- **Reproduce experiments**

    In the experiments, we used TID2013, which can be downloaded at https://www.ponomarenko.info/tid2013.htm
    
    The images from JPEG, BPG, HiFiC compressors are from the demo webpage of HiFiC https://hific.github.io/
    
## SCL API
- To use the `ImageQualityMetrics` API in SCL, first instantiate an `ImageQualityMetrics` object
    ```
    metrics = ImageQualityMetrics()
    ```
- Use
    ```
    metrics.evaluate(img0, img1)
    ```
    to evaluate the metrics between the original `img0` and the compressed `img1`. Note that both images should be NumPy array with the dimension `(H, W, C)`
    
- Use
    ```
    metrics_dict = metrics.get_metrics()
    ```
    to get a python dictionary of PSNR, SSIM, MS-SSIM, VIF, and LPIPS. Use the lower case to access the results, e.g. `metrics[ms-ssim]`. 
    
    Use
    ```
    metrics_txt = metrics.get_metrics_text()
    ```
    To get a printable string of the results.
    ```
    >>> print(metrics_txt)
    >>> PSNR: 31.156191733985963
        SSIM: 2.9743873578712554
        MS-SSIM: 0.9898914694786072
        VIF: 1.0224043172647428
        LPIPS:0.08465027064085007
    ```
    

