"""Experimental GPU backend for the original rigid RANSAC objective.

Validated separately before use. Samples use MT19937; distances and transforms
use float64. The pinned Open3D backend remains available as the control.
"""
import hashlib
import math
import numpy as np
import torch
import open3d as o3d
from torch.utils.cpp_extension import load_inline

CPP = r'''
#include <torch/extension.h>
#include <random>
torch::Tensor sample_indices(int64_t seed, int64_t size, int64_t count) {
    std::mt19937 engine(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int64_t> distribution(0, size - 1);
    auto result = torch::empty({count, 3}, torch::TensorOptions().dtype(torch::kInt64));
    auto data = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < count * 3; ++i) data[i] = distribution(engine);
    return result;
}
torch::Tensor nearest_cuda(torch::Tensor source, torch::Tensor target,
    torch::Tensor transforms, torch::Tensor origin, torch::Tensor offsets,
    int64_t nx, int64_t ny, int64_t nz, double radius);
'''

CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>

__global__ void nearest_kernel(const double* src, const double* ref,
    const double* matrices, const double* origin, const int64_t* offsets,
    int nx, int ny, int nz, double radius, int64_t n, int64_t total, double* out) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const double* p = src + (index % n) * 3;
    const double* t = matrices + (index / n) * 16;
    double x = t[0]*p[0] + t[1]*p[1] + t[2]*p[2] + t[3];
    double y = t[4]*p[0] + t[5]*p[1] + t[6]*p[2] + t[7];
    double z = t[8]*p[0] + t[9]*p[1] + t[10]*p[2] + t[11];
    int ix = static_cast<int>(floor((x-origin[0])/radius));
    int iy = static_cast<int>(floor((y-origin[1])/radius));
    int iz = static_cast<int>(floor((z-origin[2])/radius));
    double best = DBL_MAX;
    if (ix < -1 || iy < -1 || iz < -1 || ix > nx || iy > ny || iz > nz) {
        out[index] = best; return;
    }
    for (int a=max(0,ix-1); a<=min(nx-1,ix+1); ++a)
      for (int b=max(0,iy-1); b<=min(ny-1,iy+1); ++b)
        for (int c=max(0,iz-1); c<=min(nz-1,iz+1); ++c) {
          int64_t key = (static_cast<int64_t>(a)*ny+b)*nz+c;
          for (int64_t j=offsets[key]; j<offsets[key+1]; ++j) {
            const double* q = ref + j*3;
            double dx=x-q[0], dy=y-q[1], dz=z-q[2];
            double d=dx*dx+dy*dy+dz*dz;
            best = d<best ? d : best;
          }
        }
    out[index] = best;
}

torch::Tensor nearest_cuda(torch::Tensor source, torch::Tensor target,
    torch::Tensor transforms, torch::Tensor origin, torch::Tensor offsets,
    int64_t nx, int64_t ny, int64_t nz, double radius) {
    TORCH_CHECK(source.is_cuda() && target.is_cuda() && transforms.is_cuda());
    TORCH_CHECK(source.scalar_type()==torch::kFloat64 && target.scalar_type()==torch::kFloat64);
    TORCH_CHECK(transforms.scalar_type()==torch::kFloat64 && origin.scalar_type()==torch::kFloat64);
    TORCH_CHECK(offsets.scalar_type()==torch::kInt64 && offsets.is_cuda());
    TORCH_CHECK(source.is_contiguous() && target.is_contiguous() && transforms.is_contiguous());
    int64_t n=source.size(0), total=n*transforms.size(0);
    auto result = torch::empty({transforms.size(0), n}, source.options());
    nearest_kernel<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        source.data_ptr<double>(), target.data_ptr<double>(), transforms.data_ptr<double>(),
        origin.data_ptr<double>(), offsets.data_ptr<int64_t>(), nx, ny, nz, radius,
        n, total, result.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
'''

_extension = None
_reference = {}


def extension():
    global _extension
    if _extension is None:
        _extension = load_inline(name='reg2inv_ransac_distance_v1', cpp_sources=CPP,
                                 cuda_sources=CUDA,
                                 functions=['sample_indices', 'nearest_cuda'],
                                 extra_cflags=['-O3'], extra_cuda_cflags=['-O3'],
                                 with_cuda=True, verbose=True)
    return _extension


def reference_grid(target, radius):
    target = np.asarray(target, np.float64)
    key = (hashlib.sha256(target.tobytes()).hexdigest(), float(radius))
    if key not in _reference:
        origin = target.min(0)
        cell = np.floor((target-origin)/radius).astype(np.int64)
        dims = cell.max(0) + 1
        assert np.prod(dims) < 10_000_000, 'Grid exceeds bounded memory budget'
        keys = (cell[:, 0]*dims[1]+cell[:, 1])*dims[2]+cell[:, 2]
        order = np.argsort(keys, kind='stable')
        offsets = np.searchsorted(keys[order], np.arange(int(np.prod(dims))+1))
        _reference[key] = (torch.from_numpy(target[order].copy()).cuda(),
                           torch.from_numpy(origin).cuda(),
                           torch.from_numpy(offsets).cuda(), tuple(map(int, dims)))
    return _reference[key]


def transforms_from_samples(source, target, correspondence, samples, cpu_context=None):
    pairs = correspondence[samples]
    x, y = source[pairs[..., 0]], target[pairs[..., 1]]
    xc, yc = x.mean(1), y.mean(1)
    covariance = (x-xc[:, None]).transpose(1, 2) @ (y-yc[:, None]) / 3.
    u, singular, vh = torch.linalg.svd(covariance)
    v = vh.transpose(1, 2)
    correction = torch.ones((len(samples), 3), device='cuda', dtype=torch.float64)
    correction[:, 2] = torch.where(torch.linalg.det(v @ u.transpose(1, 2)) < 0, -1., 1.)
    rotation = (v*correction[:, None]) @ u.transpose(1, 2)
    matrices = torch.eye(4, device='cuda', dtype=torch.float64)[None].repeat(len(samples), 1, 1)
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = yc - (rotation @ xc[..., None]).squeeze(-1)
    # Rank-one/zero samples have multiple optimal rotations. Preserve Open3D's
    # choice in these cases rather than relying on a different SVD null basis.
    degenerate = torch.where(singular[:, 1] <= 1e-12 * singular[:, 0].clamp_min(1e-15))[0]
    if len(degenerate):
        if cpu_context is None:
            cpu_context = make_cpu_context(source.cpu().numpy(), target.cpu().numpy(),
                                           correspondence.cpu().numpy())
        src_cloud, ref_cloud, corr, estimator = cpu_context
        triples = samples[degenerate].cpu().numpy()
        native = np.stack([estimator.compute_transformation(
            src_cloud, ref_cloud, o3d.utility.Vector2iVector(corr[sample]))
            for sample in triples])
        matrices[degenerate] = torch.as_tensor(native, dtype=torch.float64, device='cuda')
    transforms_from_samples.last_degenerate = len(degenerate)
    return matrices.contiguous()


def make_cpu_context(src, ref, corr):
    clouds = []
    for points in (src, ref):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
        clouds.append(cloud)
    return (*clouds, np.asarray(corr), o3d.pipelines.registration.TransformationEstimationPointToPoint())


def evaluate(source, target_np, matrices, radius):
    ref, origin, offsets, dims = reference_grid(target_np, radius)
    distances = extension().nearest_cuda(source.contiguous(), ref, matrices.contiguous(),
                                         origin, offsets, *dims, float(radius))
    valid = distances < radius*radius
    counts = valid.sum(1)
    error2 = torch.where(valid, distances, 0.).sum(1)
    rmse = torch.where(counts > 0, (error2/counts.clamp_min(1)).sqrt(), 0.)
    return counts, rmse


@torch.inference_mode()
def register(src_points, ref_points, correspondences=None, distance_threshold=.05,
             ransac_n=3, num_iterations=10000, batch_size=50000):
    assert ransac_n == 3, 'Only original three-point rigid RANSAC is supported'
    assert 1 <= batch_size <= 50000
    # Bound distance+mask/reduction temporaries separately from model memory.
    batch_size = min(batch_size, max(1, 64_000_000 // max(1, len(src_points))))
    if correspondences is None:
        indices = np.arange(len(src_points))
        correspondences = np.stack((indices, indices), 1)
    if len(correspondences) < 3 or distance_threshold <= 0:
        register.last = dict(evaluated_hypotheses=0, convergence_budget=0,
                             fitness=0., rmse=0., degenerate_cpu_fallback=0,
                             seed=int(torch.initial_seed()), backend='experimental_gpu_ransac',
                             reason='insufficient_correspondences_or_invalid_radius')
        return np.eye(4)
    source = torch.as_tensor(np.asarray(src_points), device='cuda', dtype=torch.float64)
    target = torch.as_tensor(np.asarray(ref_points), device='cuda', dtype=torch.float64)
    corr = torch.as_tensor(correspondences, device='cuda', dtype=torch.int64)
    samples = extension().sample_indices(int(torch.initial_seed()), len(corr), num_iterations).cuda()
    cpu_context = make_cpu_context(src_points, ref_points, correspondences)
    budget, best_count, best_rmse = num_iterations, 0, 0.
    best = torch.eye(4, device='cuda', dtype=torch.float64)
    evaluated = 0
    degenerate_fallback = 0
    for begin in range(0, num_iterations, batch_size):
        if begin >= budget:
            break
        end = min(begin+batch_size, budget)
        matrices = transforms_from_samples(source, target, corr, samples[begin:end], cpu_context)
        degenerate_fallback += transforms_from_samples.last_degenerate
        counts, rmse = evaluate(source, ref_points, matrices, distance_threshold)
        cpu_counts, cpu_rmse = counts.cpu().tolist(), rmse.cpu().tolist()
        candidate_indices = []
        count, error = best_count, best_rmse
        for i, (n, e) in enumerate(zip(cpu_counts, cpu_rmse)):
            if n > count or (n == count and e < error):
                candidate_indices.append(i)
                count, error = n, e
        if candidate_indices:
            transforms = matrices[candidate_indices]
            transformed = source[corr[:, 0]][None] @ transforms[:, :3, :3].transpose(1, 2)
            transformed += transforms[:, None, :3, 3]
            ratios = ((transformed-target[corr[:, 1]][None]).square().sum(-1) <
                      distance_threshold**2).double().mean(1).cpu().tolist()
            for i, ratio in zip(candidate_indices, ratios):
                if begin+i >= budget:
                    break
                best_count, best_rmse = cpu_counts[i], cpu_rmse[i]
                best = matrices[i].clone()
                if ratio >= 1.:
                    budget = 1
                elif ratio > 0.:
                    estimate = math.ceil(math.log(1.-.9999)/math.log1p(-ratio**3))
                    budget = min(budget, estimate)
        evaluated += len(matrices)
    register.last = dict(evaluated_hypotheses=evaluated, convergence_budget=budget,
                         batch_size=batch_size,
                         fitness=best_count/len(source), rmse=best_rmse,
                         degenerate_cpu_fallback=degenerate_fallback,
                         seed=int(torch.initial_seed()), backend='experimental_gpu_ransac')
    return best.cpu().numpy()
