#!/usr/bin/env python3
"""
RKNN 模型 benchmark 工具
用法: python3 benchmark_rknn.py <model.rknn> [--runs 50]
"""
import argparse
import ctypes
import time
import numpy as np

# ── RKNN C 结构体 ─────────────────────────────────────────────────────────

class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [('n_input', ctypes.c_uint32), ('n_output', ctypes.c_uint32)]

class RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ('index', ctypes.c_uint32), ('n_dims', ctypes.c_uint32),
        ('dims', ctypes.c_uint32 * 16), ('name', ctypes.c_char * 256),
        ('n_elems', ctypes.c_uint32), ('size', ctypes.c_uint32),
        ('fmt', ctypes.c_int32), ('type', ctypes.c_int32),
        ('qnt_type', ctypes.c_int32), ('fl', ctypes.c_int8),
        ('zp', ctypes.c_int32), ('scale', ctypes.c_float),
        ('w_stride', ctypes.c_uint32), ('size_with_stride', ctypes.c_uint32),
        ('pass_through', ctypes.c_uint8), ('h_stride', ctypes.c_uint32),
    ]

class RknnInput(ctypes.Structure):
    _fields_ = [
        ('index', ctypes.c_uint32), ('buf', ctypes.c_void_p),
        ('size', ctypes.c_uint32), ('pass_through', ctypes.c_uint8),
        ('type', ctypes.c_int32), ('fmt', ctypes.c_int32),
    ]

class RknnOutput(ctypes.Structure):
    _fields_ = [
        ('want_float', ctypes.c_uint8), ('is_prealloc', ctypes.c_uint8),
        ('index', ctypes.c_uint32), ('buf', ctypes.c_void_p),
        ('size', ctypes.c_uint32),
    ]

class RknnPerfRun(ctypes.Structure):
    _fields_ = [('run_duration', ctypes.c_int64)]

class RknnSdkVersion(ctypes.Structure):
    _fields_ = [('api_version', ctypes.c_char * 256), ('drv_version', ctypes.c_char * 256)]

# ── 常量 ──────────────────────────────────────────────────────────────────
RKNN_SUCC              = 0
RKNN_QUERY_IN_OUT_NUM  = 0
RKNN_QUERY_INPUT_ATTR  = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_RUN    = 4
RKNN_QUERY_SDK_VERSION = 5
RKNN_TENSOR_UINT8      = 3
RKNN_TENSOR_NHWC       = 1
RKNN_NPU_CORE_0_1_2    = 7


def main():
    parser = argparse.ArgumentParser(description='RKNN 模型 benchmark')
    parser.add_argument('model', help='rknn 模型文件路径')
    parser.add_argument('--runs', type=int, default=50, help='推理次数 (默认50)')
    parser.add_argument('--cores', type=int, default=7,
                        help='NPU core mask: 1=core0, 3=0+1, 7=0+1+2 (默认7)')
    args = parser.parse_args()

    lib = ctypes.CDLL('/usr/lib/librknnrt.so')

    # 设置函数原型
    lib.rknn_init.restype  = ctypes.c_int
    lib.rknn_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    lib.rknn_query.restype  = ctypes.c_int
    lib.rknn_query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    lib.rknn_set_core_mask.restype  = ctypes.c_int
    lib.rknn_set_core_mask.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.rknn_inputs_set.restype  = ctypes.c_int
    lib.rknn_inputs_set.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    lib.rknn_run.restype  = ctypes.c_int
    lib.rknn_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.rknn_outputs_get.restype  = ctypes.c_int
    lib.rknn_outputs_get.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                     ctypes.c_void_p, ctypes.c_void_p]
    lib.rknn_outputs_release.restype  = ctypes.c_int
    lib.rknn_outputs_release.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    lib.rknn_destroy.restype  = ctypes.c_int
    lib.rknn_destroy.argtypes = [ctypes.c_void_p]

    def query(ctx, cmd, info):
        ret = lib.rknn_query(ctx, cmd, ctypes.byref(info), ctypes.sizeof(info))
        if ret != RKNN_SUCC:
            raise RuntimeError(f'rknn_query({cmd}) failed: {ret}')

    # 加载模型
    print(f'Loading: {args.model}')
    with open(args.model, 'rb') as f:
        model_data = f.read()
    buf = ctypes.create_string_buffer(model_data)
    ctx = ctypes.c_void_p()
    ret = lib.rknn_init(ctypes.byref(ctx), buf, len(model_data), 0, None)
    if ret != RKNN_SUCC:
        raise RuntimeError(f'rknn_init failed: {ret}')

    # SDK 版本
    ver = RknnSdkVersion()
    query(ctx, RKNN_QUERY_SDK_VERSION, ver)
    api_str = ver.api_version.split(b'\0')[0].decode()
    drv_str = ver.drv_version.split(b'\0')[0].decode()
    print(f'SDK api={api_str} driver={drv_str}')

    # NPU core mask
    ret = lib.rknn_set_core_mask(ctx, args.cores)
    print(f'Core mask: {args.cores} → {"OK" if ret==0 else "FAILED"}')

    # 输入/输出数量
    io = RknnInputOutputNum()
    query(ctx, RKNN_QUERY_IN_OUT_NUM, io)
    print(f'Tensors: inputs={io.n_input} outputs={io.n_output}')

    # 输入属性
    input_dims = None
    for i in range(io.n_input):
        attr = RknnTensorAttr(index=i)
        query(ctx, RKNN_QUERY_INPUT_ATTR, attr)
        dims = list(attr.dims[:attr.n_dims])
        name = attr.name.split(b'\0')[0].decode(errors='replace')
        print(f'  Input[{i}]  name={name} dims={dims} fmt={attr.fmt} type={attr.type} '
              f'zp={attr.zp} scale={attr.scale:.6g} size={attr.size}')
        if i == 0:
            input_dims = dims  # e.g. [1, H, W, 3] NHWC

    for i in range(io.n_output):
        attr = RknnTensorAttr(index=i)
        query(ctx, RKNN_QUERY_OUTPUT_ATTR, attr)
        dims = list(attr.dims[:attr.n_dims])
        name = attr.name.split(b'\0')[0].decode(errors='replace')
        print(f'  Output[{i}] name={name} dims={dims} fmt={attr.fmt} type={attr.type} '
              f'zp={attr.zp} scale={attr.scale:.6g} size={attr.size}')

    # 确定输入尺寸
    if input_dims and len(input_dims) >= 3:
        # NHWC → [N, H, W, C]
        infer_h = input_dims[1]
        infer_w = input_dims[2]
        infer_c = input_dims[3] if len(input_dims) >= 4 else 3
    else:
        infer_h, infer_w, infer_c = 416, 416, 3
    print(f'\nInput size: {infer_h}x{infer_w}x{infer_c}')

    # 生成随机输入
    dummy = np.random.randint(0, 255, (infer_h, infer_w, infer_c), dtype=np.uint8)
    dummy = np.ascontiguousarray(dummy)

    # ── benchmark ────────────────────────────────────────────────────────
    print(f'\nRunning {args.runs} inferences (warmup=5)...')
    npu_times = []
    wall_times = []

    for i in range(args.runs + 5):
        inp = RknnInput(
            index=0,
            buf=dummy.ctypes.data_as(ctypes.c_void_p),
            size=dummy.nbytes,
            pass_through=0,
            type=RKNN_TENSOR_UINT8,
            fmt=RKNN_TENSOR_NHWC,
        )
        ret = lib.rknn_inputs_set(ctx, 1, ctypes.byref(inp))
        if ret != RKNN_SUCC:
            raise RuntimeError(f'rknn_inputs_set failed: {ret}')

        t0 = time.perf_counter()
        ret = lib.rknn_run(ctx, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f'rknn_run failed: {ret}')
        wall_ms = (time.perf_counter() - t0) * 1000

        outputs = (RknnOutput * io.n_output)()
        for j, o in enumerate(outputs):
            o.want_float = 1
            o.index = j
        ret = lib.rknn_outputs_get(ctx, io.n_output, outputs, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f'rknn_outputs_get failed: {ret}')
        lib.rknn_outputs_release(ctx, io.n_output, outputs)

        perf = RknnPerfRun()
        ret = lib.rknn_query(ctx, RKNN_QUERY_PERF_RUN, ctypes.byref(perf), ctypes.sizeof(perf))
        npu_us = perf.run_duration if ret == RKNN_SUCC else 0

        if i >= 5:  # 跳过 warmup
            npu_times.append(npu_us / 1000.0)  # us → ms
            wall_times.append(wall_ms)

    lib.rknn_destroy(ctx)

    npu_arr  = np.array(npu_times)
    wall_arr = np.array(wall_times)

    print('\n===== Benchmark Results =====')
    print(f'Model:      {args.model}')
    print(f'Input:      {infer_h}x{infer_w}')
    print(f'Runs:       {args.runs}')
    print()
    print(f'NPU time    avg={npu_arr.mean():.1f}ms  min={npu_arr.min():.1f}ms  '
          f'max={npu_arr.max():.1f}ms  p95={np.percentile(npu_arr, 95):.1f}ms')
    print(f'Wall time   avg={wall_arr.mean():.1f}ms  min={wall_arr.min():.1f}ms  '
          f'max={wall_arr.max():.1f}ms  p95={np.percentile(wall_arr, 95):.1f}ms')
    print()
    max_fps_npu  = 1000.0 / npu_arr.mean()  if npu_arr.mean()  > 0 else 0
    max_fps_wall = 1000.0 / wall_arr.mean() if wall_arr.mean() > 0 else 0
    print(f'Max FPS (NPU):  {max_fps_npu:.1f}')
    print(f'Max FPS (wall): {max_fps_wall:.1f}')
    print()
    if max_fps_wall >= 30:
        print('结论: 可以达到 30fps ✓')
    elif max_fps_wall >= 20:
        print(f'结论: 最多约 {max_fps_wall:.0f}fps，无法稳定 30fps')
    else:
        print(f'结论: 较慢，约 {max_fps_wall:.0f}fps，建议保留 416×416 模型')


if __name__ == '__main__':
    main()
