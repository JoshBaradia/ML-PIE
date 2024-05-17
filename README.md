# We propose an attention-guided UNet leveraging spatial/channel attention and feature pyramid integration to correct color and restore details in underwater and also Enhansing the resolution of any Underwater Image using GANs.

## Code will be updated once the paper is published :) 

## ✨UFPN

### 🐳UFPN Training

## ✨ESRGAN

Enhancing images upto 4x times. Using UFPN output images as input to our ESRGAN model.

### 🐳ESRGAN Training
**Make sure that all the dependencies are satisfield use `requirements.txt` file to load the required dependencies.**
```bash
pip install -r requirements.txt
```

- Use the file `/options/train_test_dataset_4x.yml` to Fine-Tune the ESRGAN model which is `RRDB_ESRGAN_x4.pth` model. 
- The new fine-tuned model gets saved.
- You can find the model in `experiments/pretrained_models/` folder. 

### 🐳Testing UFPN+ESRGAN
**Using our best UFPN model `checkpoint_trial_37.pth` and `net_g_240000.pth`**
- Run the `test_test.py` python script. The UFPN model takes input, which are the Low Resolution-Degraded images from `/testrun_enet` folder and the script saves it in `/inputs` folder. 
- Then our testing script `inference_realesrgan.py` uses our ESRGAN model, takes input images from the `/inputs` folder which now has enhanced images and saves it in `/results` folder.
- Thus the `/results` folder contains enhanced-super-resolution images. 

```bash
python test_test.py;  python inference_realesrgan.py --model_path experiments/pretrained_models/net_g_240000.pth --input inputs
```
