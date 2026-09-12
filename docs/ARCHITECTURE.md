# Current implementation and proposed paper design

## Current implementation

The deployed DDP checkpoint has two client paths.

- **G-Path:** a VGG11-BN client cut after block 4 produces a `16 x 16 x 16`
  spatial feature map. Gaussian noise is added before transmission. Slot-CEM is
  a training regulariser over same-identity examples and a bounded memory bank;
  it does not compress the transmitted map into prototypes.
- **S-Path:** an ImageNet-pretrained MobileNetV3-Large is followed by global
  average pooling and `Linear -> LayerNorm -> GELU -> Dropout`, producing a
  256-dimensional token. A separate Gaussian channel is applied before release.
- **Fusion:** the server computes
  `(1 - alpha) * g_logits / T_g + alpha * s_logits / T_s`, where `alpha`, `T_g`,
  and `T_s` are global learned parameters. This is calibrated logit fusion, not
  an input-dependent gate.
- **Training:** stage 1 trains the semantic path and fusion while the spatial
  model is fixed. Stage 2 keeps the spatial client fixed while adapting the
  spatial server, semantic path, and fusion to the stronger deployment noise.

## Slot-CEM foundation

For each identity, the spatial pretraining implementation combines current
features with at most 64 stored historical features. It projects the flattened
4096-dimensional feature to 64 dimensions for soft assignment to 8 slots over
3 iterations. The privacy regulariser is based on soft within-slot geometric
variance. The assignment is across images of the same identity, not across
spatial positions inside one image.

## Paper design discussed with the supervisor

The supervisor's revised scientific framing considers a more explicit
geometry/semantic separation, prototype-style transmission, and a potentially
dynamic fusion mechanism. Those descriptions are not yet the current code.
They should be treated as design candidates until the authors agree on one
architecture and implement it. Existing DDP numbers must not be attributed to
an unimplemented variant.

## Privacy target

Identity prediction is the downstream task, so the system cannot claim to hide
the identity label from the server. The privacy target is reconstruction of the
input image and its non-task visual detail, including appearance, pose, and
background. Both released tensors are part of the attack surface.
