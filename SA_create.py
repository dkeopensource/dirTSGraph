import os
import argparse

import numpy as np

from src.utils import read_X

#import dtw
import csv
import gc
import psutil

import torch
import torch.nn.functional as F

dataset_dir = './datasets/UCRArchive_2018'
output_dir = './attention'

def argsparser():
	parser = argparse.ArgumentParser("Sef-attention computation")
	parser.add_argument('--dataset', help='Dataset name', default='Coffee')

	return parser

def pad_to_maxlen(X):
	"""Pad variable-length list/array X into (N, T_max) and create a corresponding mask (N, T_max)."""
	N = len(X)
	lens = np.array([len(x) for x in X], dtype=np.int32)
	T = int(lens.max())
	X_pad = np.zeros((N, T), dtype=np.float32)
	mask = np.zeros((N, T), dtype=bool)
	for i, x in enumerate(X):
		L = lens[i]
		X_pad[i, :L] = x
		mask[i, :L] = True
	return X_pad, mask, lens, T


def compute_attention_E(attn_ab, attn_ba):
	"""
	attn_ab: attention (a → b)
	attn_ba: attention (b → a)
	"""
	s = attn_ab + attn_ba
	if s == 0:
		return 0.0
	else:
		direction_term = attn_ab / s
		strength_term = np.log1p(s)  # or np.sqrt(s)
		return direction_term * strength_term

def compute_attention_J(attn_ab, attn_ba):
	denom = attn_ab + attn_ba
	if denom == 0:
		return 0.0
	else:
		return (attn_ab * 2) / denom  # emphasize ratio

def get_self_attention_memmap_batch_safe(
	X, out_path_sa, batch_size=64, sub_batch_size=32, tau=0.1, device='cuda', print_all=True
):
	"""
	Compute a self-attention-based "distance" matrix (instead of DTW) using memmap.
	- X: list/ndarray of 1D time series (variable length supported)
	- out_path: np.memmap save path (file prefix; without extension)
	"""
	# (1) Padding + normalization (z-score per time series; NaN -> 0)
	Xp, mask_np, lens, T = pad_to_maxlen(X)
	Xp = np.nan_to_num(Xp, nan=0.0)
	# Per-series z-score normalization
	mu = Xp.mean(axis=1, keepdims=True)
	std = Xp.std(axis=1, keepdims=True) + 1e-8
	Xp = (Xp - mu) / std

	# (2) Convert to torch tensors
	x_all = torch.tensor(Xp, dtype=torch.float32, device=device)   # (N, T)
	m_all = torch.tensor(mask_np, dtype=torch.bool, device=device) # (N, T)
	N = x_all.size(0)

	# (3) Prepare memory allocation
	distances = np.zeros((N, N), dtype=np.float64)

	# (4) Temporal feature projection (Q, K) — linear projection without training; d_model=32
	d_model = 32
	# Fixed (non-trainable) projection: use fixed random seed for reproducibility
	torch.manual_seed(0)
	Wq = F.normalize(torch.randn(1, d_model, device=device), dim=1)  # (1, d)
	Wk = F.normalize(torch.randn(1, d_model, device=device), dim=1)  # (1, d)

	def time_feature(z):  # (B, T) -> (B, T, d)
		# Simple feature expansion: concatenate [z, Δz, pos] then project to d_model
		# When d_model=32, this behaves like a 3-channel linear combination (simplified as 1x1 projection)
		dz = F.pad(z[:, 1:] - z[:, :-1], (1, 0))	# (B, T)
		pos = torch.linspace(-1, 1, z.size(1), device=device).unsqueeze(0).expand_as(z)
		feat = torch.stack([z, dz, pos], dim=-1)	# (B, T, 3)
		# Simplify: combine 3 channels with random weights to scalar, then expand to d_model
		w3 = F.normalize(torch.randn(3, device=device), dim=0)
		s = (feat * w3).sum(-1, keepdim=True)	   # (B, T, 1)
		# Expand to d_model: (B, T, 1) * W -> (B, T, d)
		return s * Wq  # arbitrary projection (shared for Q/K; replaced with Wk below)

	scale = d_model ** 0.5

	for batch_start in range(0, N, batch_size):
		batch_end = min(batch_start + batch_size, N)
		B = batch_end - batch_start

		xb = x_all[batch_start:batch_end]   # (B, T)
		mb = m_all[batch_start:batch_end]   # (B, T)

		# Q: (B, T, d), K/V: (N, T, d)
		Q = time_feature(xb) / scale					# (B, T, d)
		K = time_feature(x_all).mul(Wk / Wq) / scale	# (N, T, d)  # different projection from Q

		# Mask
		mbf = mb.float()		   # (B, T)
		maf = m_all.float()		# (N, T)

		# For efficient computation of cross-attention scores, block the process (sub_batch_size = B_sub)
		# Direct computation of (B_sub, T, d) × (N, T, d)^T → (B_sub, N, T, T) is memory expensive.
		# Instead, compute a "soft alignment score" per row:
		# s_{i→j} = mean_t softmax_tau(Q_i[t]·K_j^T)[valid_j]
		# Implementation note: (B_sub, T, d) @ (N, d, T) → (B_sub, N, T, T) is skipped;
		# reduce to (B_sub, N, T) with row-wise softmax and mean.

		# Transpose K to (N, d, T)
		KT = K.transpose(1, 2).contiguous()  # (N, d, T)

		# Memory-efficient sub-batching
		block = np.zeros((B, N), dtype=np.float64)

		for sub_start in range(0, B, sub_batch_size):
			sub_end = min(sub_start + sub_batch_size, B)
			bs = sub_end - sub_start

			Q_sub = Q[sub_start:sub_end]		# (bs, T, d)
			mb_sub = mbf[sub_start:sub_end]	 # (bs, T)

			# (bs, T, d) @ (N, d, T) -> (bs, N, T, T)
			# Instead of constructing the full tensor, compute per-time scores (bs, N, T),
			# apply softmax, then take the mean.
			logits = torch.einsum('btd,ndt->bnt', Q_sub, KT)  # (bs, N, T)

			# Mask: consider only valid positions in target j along T axis
			logits = logits / tau
			logits = logits.masked_fill(~maf.unsqueeze(0).expand_as(logits).bool(), float('-inf'))

			att = torch.softmax(logits, dim=2)  # (bs, N, T), softmax along time axis per (b, n)

			# Average only over valid query time steps (mb_sub)
			# s_{i→j} = (1/|Ti|) Σ_t [ Σ_s att[b,n,s] * 1 ] = (1/|Ti|) Σ_t 1  (att sums to 1 over time)
			# This would yield constant 1, so use "concentration" as a meaningful quantitative measure.
			# Example: entropy-based concentration — lower entropy → higher score.
			eps = 1e-12
			H = -(att * (att.clamp_min(eps).log())).sum(dim=2)  # (bs, N)
			# Valid length of target j
			Lj = maf.sum(dim=1).clamp_min(1.0)				  # (N,)
			H_norm = H / (Lj.log())							 # (bs, N)
			# Score increases with higher concentration → 1 - normalized entropy
			s_i2j = (1.0 - H_norm).clamp_min(0.0)			   # (bs, N)

			# Optionally approximate s_{j→i} for symmetry (omitted for efficiency)
			block[sub_start:sub_end, :] = s_i2j.detach().cpu().numpy()

		# Optional symmetrization: s_ij = (s_i2j + s_j2i) / 2
		# Omitted here since only one direction is used.

		# Convert similarity to distance: d = -log(s + eps)
		eps = 1e-6
		dist_block = -np.log(block + eps)

		# Diagonal = 0, symmetric copy
		for bi, i in enumerate(range(batch_start, batch_end)):
			dist_block[bi, i] = 0.0

		# Write block to memory
		distances[batch_start:batch_end, :] = dist_block

	N = len(distances)
	attn_E = np.zeros((N, N))

	for i in range(N):
		for j in range(N):
			if i != j:
				attn_E[i][j] = compute_attention_E(distances[i][j], distances[j][i])

	# (10) Save results (.npy automatically appended)
	if not out_path_sa.endswith('.npy'):
		out_path_sa += '.npy'
	np.save(out_path_sa, attn_E)

	del distances

if __name__ == "__main__":
	# Get the arguments
	parser = argsparser()
	args = parser.parse_args()

	result_dir = os.path.join(output_dir, 'ucr_datasets_attention')

	result_dir_sa = os.path.join(output_dir, 'ucr_datasets_attention_sa')

	if not os.path.exists(result_dir_sa):
		os.makedirs(result_dir_sa)


	X = read_X(dataset_dir, args.dataset)
	
	out_path_sa = os.path.join(result_dir_sa, args.dataset)

	tau = 0.01

	# Batch processing
	get_self_attention_memmap_batch_safe(
		X, out_path_sa,
		batch_size=64, sub_batch_size=32,
		tau=tau, device='cuda' if torch.cuda.is_available() else 'cpu', print_all=False
	)
	
	'''
	print(f"Finished {args.dataset}")
	'''
