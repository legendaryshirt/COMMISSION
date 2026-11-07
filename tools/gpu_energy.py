#!/usr/bin/env python3
"""Print the GPU's cumulative energy counter in millijoules (NVML, Volta and newer)."""
import ctypes
nvml = ctypes.CDLL("libnvidia-ml.so.1")
assert nvml.nvmlInit_v2() == 0
dev = ctypes.c_void_p()
assert nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev)) == 0
mj = ctypes.c_ulonglong()
assert nvml.nvmlDeviceGetTotalEnergyConsumption(dev, ctypes.byref(mj)) == 0
print(mj.value)
