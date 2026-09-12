# Dataset setup

## FaceScrub

FaceScrub is distributed as image URLs and metadata. This repository does not
redistribute the images. Obtain the data under the dataset's terms from one of
the maintained source pages:

- Dataset information: https://malea.winkler.site/facescrub.html
- Public downloader and CC BY-NC 4.0 notice:
  https://github.com/lightalchemist/FaceScrub
- MegaFace access page, which also documents FaceScrub access conditions:
  https://megaface.cs.washington.edu/participate/challenge.html

The DDP runners expect an ImageFolder layout:

```text
facescrub_official/
  train/
    identity_000/
      image_*.jpg
    ...
  val/
    identity_000/
      image_*.jpg
    ...
```

The processed dataset used for the current evidence contains 45,760 images:
41,425 in training and 4,335 in validation, with 526 identities present. The
legacy classifier has 530 outputs because four classes from the original
mapping are absent after processing. Preserve `class_to_idx` across splits.

Use the preparation utilities under `archive/current_artefact/src/` and then
record the split and file hashes before training. Do not silently replace
missing URLs or change the split for a matched CEM comparison.

## CIFAR-10 and CIFAR-100

Cross-dataset configurations are provided in `configs/`. Torchvision can
download these datasets into an ignored local directory:

```bash
python scripts/build_torchvision_protocol.py --help
```

The exact target backbone, cut layer, training budget, and attack protocol must
be frozen with the authors before these runs are used in the paper.

## Private checkpoint transfer

The current G-Path foundation requires three legacy files:

```text
checkpoint_f_best.tar
checkpoint_cloud_best.tar
checkpoint_classifier_best.tar
```

Transfer them directly to the collaborator through an institution-approved
private channel. Send a SHA-256 manifest separately and verify it before any
run. Public Google Drive links are not recommended for face data or unpublished
checkpoints.
