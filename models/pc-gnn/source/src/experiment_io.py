import csv
import datetime
import json
import os
import platform
import sys

import numpy as np
import torch


def utc_now_iso():
	return datetime.datetime.now(datetime.timezone.utc).isoformat()


def synchronise_device(use_cuda):
	if use_cuda:
		torch.cuda.synchronize()


def json_safe(value):
	if isinstance(value, dict):
		return {str(key): json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [json_safe(item) for item in value]
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	return str(value)


def write_json(path, payload):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as handle:
		json.dump(json_safe(payload), handle, indent=2, sort_keys=True)


def write_csv(path, rows, fieldnames):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({field: row.get(field, '') for field in fieldnames})


def environment_metadata(use_cuda):
	metadata = {
		'python_version': sys.version,
		'platform': platform.platform(),
		'processor': platform.processor(),
		'python_executable': sys.executable,
		'numpy_version': np.__version__,
		'torch_version': torch.__version__,
		'cuda_available': bool(torch.cuda.is_available()),
		'cuda_used': bool(use_cuda),
		'cuda_version': torch.version.cuda,
	}
	if use_cuda:
		metadata['gpu_name'] = torch.cuda.get_device_name(0)
		metadata['gpu_total_memory_bytes'] = int(torch.cuda.get_device_properties(0).total_memory)
	else:
		metadata['gpu_name'] = None
		metadata['gpu_total_memory_bytes'] = None
	return metadata
