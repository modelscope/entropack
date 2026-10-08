# EntroPack lattice compression

EntroPack's lattice scheme combines lossy vector quantization with lossless entropy coding
to compress two-dimensional tensors. `LatticeRANSConfig` accepts finite target bitrates in
[0.001, 11] bits per parameter, including non-integer values. Decompression retains the input dtype,
while the target parameter controls storage rate.

![EntroPack encoding and decoding pipeline](../../assets/entropack-pipeline.png)

EntroPack's encoding and decoding pipeline, illustrated with a weight matrix. The upper panel
shows rate search and encoding; the lower panel shows fused GPU decoding and reconstruction.

## Lattice quantization and integer fields

Rows can differ substantially in numerical scale. The encoder first normalizes each row
by its root mean square, then groups the normalized values into eight-dimensional vectors.
Each vector is approximated by its nearest point on a scaled E8 lattice, a regular
arrangement of points in eight dimensions. A shared quantization scale controls the spacing
between these points. Finer spacing generally reduces reconstruction error but requires
more bits to describe the selected points.

E8 has integer-coordinate and half-integer-coordinate subsets, called cosets. EntroPack
represents each point by its coset and eight invertible integer fields, using the lattice's
parity constraint to compact the final coordinate. The probability model conditions each
coordinate field on the coset, capturing differences between the two subsets. Frequently
occurring field values can then be encoded with fewer bits on average. The fields can
later be inverted arithmetically without a reconstruction codebook.

## Rate selection and refinement

The encoder searches for a quantization scale using sampled rows. For each candidate scale,
it estimates the coded field size and the metadata needed for decoding. This avoids
repeatedly producing a full compressed stream during the search. Once the scale is selected,
the encoder quantizes the full tensor and fits a reconstruction scale for each row by
least squares.

Optional per-row rate–distortion refinement compares several resolutions for each row and
allocates them under an estimated storage budget. It alternates candidate selection with
updates to the shared probability model. This adds encoding work and is disabled by default.

## Encoding and reconstruction

The selected fields are encoded with rANS in independently decodable tiles.
The representation also stores the probability tables, row scales, and tile metadata.
Decoding recovers the fields, reconstructs lattice points, and applies the row scales in
a fused GPU operation. The result has the input's shape and dtype.

Quantization and conversion back to the output dtype determine reconstruction error.
Entropy coding itself preserves the selected fields exactly. The achieved bitrate can
differ from the requested target because scale selection uses a size estimate. The
`actual_bpp` property reports the actual stored bytes, including metadata, divided by the
element count and multiplied by eight.

## Usage

The example targets 3.5 bits per parameter and measures relative L2 error against the input.
Search and refinement settings are described in [Config](../Usage/Configuration.md).

```python
import torch
import entropack as ep

tensor = (torch.randn(256, 256, device="cuda") * 0.02).to(torch.bfloat16)
config = ep.LatticeRANSConfig(target_bpp=3.5)

compressed = ep.compress(tensor, config)
restored = ep.decompress(compressed, config)

reference = tensor.float()
relative_l2 = (restored.float() - reference).norm() / reference.norm()
assert restored.shape == tensor.shape
assert restored.dtype == tensor.dtype
print(f"Target: {config.target_bpp:.2f} bits per parameter")
print(f"Stored: {compressed.actual_bpp:.2f} bits per parameter")
print(f"Relative L2 error: {100 * relative_l2.item():.2f}%")
```
