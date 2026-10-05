First 25 test images in frozen manifest order; 32-step Gaussian model, seed 17.
Decoding uses encrypted.npy and noise.npy (float32 CHW RGBA), not the preview image.
Encrypted PNGs clip RGB to [0,1] and hide the transformed alpha for display only.
Original/decrypted PNGs retain RGBA. Checkerboards in figures are display backgrounds.
MAE and maximum errors are measured before byte rounding; exactness compares all RGBA pixel bytes.
These examples establish numerical recovery, not confidentiality.
