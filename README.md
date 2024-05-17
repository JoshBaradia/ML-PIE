# We propose an attention-guided UNet leveraging spatial/channel attention and feature pyramid integration to correct color and restore details in underwater and also Enhansing the resolution of any Underwater Image using GANs.


## ESRGAN

Enhancing images upto 4x times. Using UFPN output images as input to our ESRGAN model.

### Training
python test_test.py: python inference realesrgan.py --model_path experiments/pretrained_ models/net_g_240000.pth -- input inputs
