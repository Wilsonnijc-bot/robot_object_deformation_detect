"""Inference-only caches for immutable CAD work; no scoring or pose changes."""
import hashlib

import torch


def install_geometry_cache():
    import model
    from geotransformer.utils.data import precompute_data_stack_mode
    original = model.precompute_data_stack_mode_two
    cache = {}
    def precompute(ref, src, ref_lengths, src_lengths, stages, voxel, radius, limits):
        key = (hashlib.sha256(ref.numpy().tobytes()).hexdigest(),
               tuple(ref_lengths.tolist()), stages, voxel, radius, tuple(limits))
        if key not in cache:
            cache[key] = precompute_data_stack_mode(ref, ref_lengths, stages, voxel, radius, limits)
        reference = cache[key]
        source = precompute_data_stack_mode(src, src_lengths, stages, voxel, radius, limits)
        return {f'{side}_{name}': data[name]
                for side, data in (('ref', reference), ('src', source))
                for name in ('points', 'neighbors', 'subsampling', 'upsampling')}
    model.precompute_data_stack_mode_two = precompute
    return lambda: setattr(model, 'precompute_data_stack_mode_two', original)


def install_backbone_cache(bank):
    backbone = bank.model.backbone
    assert not bank.model.training and not backbone.training
    reference = torch.from_numpy(bank.reference).cuda()
    original = backbone.forward
    cached = None
    hits = [0]
    def forward(points, neighbors, subsampling, upsampling):
        nonlocal cached
        is_reference = (points[0].shape == reference.shape and
                        torch.equal(points[0], reference))
        if not is_reference:
            return original(points, neighbors, subsampling, upsampling)
        if cached is None:
            result = original(points, neighbors, subsampling, upsampling)
            cached = ([x.detach().clone() for x in result[0]], result[1].detach().clone())
        else:
            hits[0] += 1
        # Return independent tensors to prevent downstream mutation of the cache.
        return [x.clone() for x in cached[0]], cached[1].clone()
    backbone.forward = forward
    def restore():
        backbone.forward = original
    return restore, hits
