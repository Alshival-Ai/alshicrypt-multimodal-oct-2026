25 held-out Pokemon: Gaussian coupling, Pokemon-only training, seed 17.
Open overview.png, sheet-01.png through sheet-05.png, or each named triplet.png.
Images were selected by manifest order, without filtering for reconstruction quality.

Encrypted PNGs are opaque RGB previews clipped to [0,1]. They are not the payload.
The actual encrypted.npy contains unclipped float32 RGBA in CHW order. Alpha is transformed.
Decoding reloads encrypted.npy and noise.npy; it never uses the preview or the original image.
Originals and byte-rounded reconstructions are shown on a checkerboard.
metrics.csv reports errors before byte rounding, including transparent/background RGB.
Exact RGBA recovery compares decoded pixels, not PNG container bytes.
The inverse is built into the architecture; these figures do not demonstrate secrecy or target-map fidelity.

Exact byte recovery: 25/25 images.
Mean normalized MAE: 7.5358459e-08; maximum error: 1.3486991e-06.
Reproduce from repository root: .venv/bin/python scripts/render_gaussian_examples.py
