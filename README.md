# We propose an attention-guided UNet leveraging spatial/channel attention and feature pyramid integration to correct color and restore details in underwater and also Enhansing the resolution of any Underwater Image using GANs.

## Code will be updated once the paper is published :)  

## ✨ESRGAN✨

Enhancing images upto 4x times. Using UFPN output images as input to our ESRGAN model.

### Training
**Make sure that all the dependencies are satisfield use `requirements.txt` file to load the required dependencies.**
```bash
pip install -r requirements.txt
```

- To Fine-Tune the ESRGAN model run the file `/options/train_test_dataset_4x.yml` which uses `RRDB_ESRGAN_x4.pth` model. 
- The new fine-tuned model gets saved.
- You can find the model in `experiments/pretrained_models/` folder. 

### Testing
**Using our best UFPN model `checkpoint_trial_37.pth` and `net_g_240000.pth`**
- The UFPN model takes input images from `testrun_enet` folder and saves it in `inputs` folder. 
- Then our ESRGAN model takes input images from the `inputs` folder and saves it in `results` folder.

```bash
python test_test.py;  python inference_realesrgan.py --model_path experiments/pretrained_models/net_g_240000.pth --input inputs
```
