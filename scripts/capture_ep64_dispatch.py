"""torchrun entry: ten fresh FP8 dispatch warmups, then one captured dispatch."""
import argparse
import json
import os
import struct
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

import deep_ep
from deep_ep.utils.math import per_token_cast_to_fp8
from deep_ep.utils.refs import dispatch as ref_dispatch


def check_output(x, idx, weights, result):
    (data, scales), _, recv_weights, handle, _ = result
    (ref_data, ref_scales), ref_idx, ref_weights, ref_src, counts = ref_dispatch(
        (x[0].view(torch.uint8), x[1]), idx, weights, 4096, 384)
    assert handle.num_recv_tokens == ref_data.size(0)
    assert torch.equal(handle.psum_num_recv_tokens_per_rank, counts.cumsum(0).int())
    meta = handle.recv_src_metadata
    order, ref_order = meta[:, -2].argsort(), ref_src.argsort()
    assert torch.equal(meta[order, -2], ref_src[ref_order])
    ref_data, ref_scales = ref_data[ref_order], ref_scales[ref_order]
    ref_idx, ref_weights = ref_idx[ref_order], ref_weights[ref_order]
    slots = meta[order, :-2]
    valid = slots >= 0
    assert torch.equal(valid, ref_idx >= 0)
    flat_slots = slots[valid].long()
    assert torch.unique(flat_slots).numel() == flat_slots.numel()
    assert flat_slots.numel() == handle.num_unaligned_recv_tokens_per_expert.sum().item()
    for actual, expected in ((data.view(torch.uint8), ref_data), (scales, ref_scales)):
        assert torch.equal(actual[flat_slots], expected[:, None, :].expand(-1, 6, -1)[valid])
    assert torch.equal(recv_weights[flat_slots], ref_weights[valid])
    padding = torch.ones(data.size(0), dtype=torch.bool, device=data.device)
    padding[flat_slots] = False
    assert not data.view(torch.uint8)[padding].any().item()
    assert not scales[padding].any().item() and not recv_weights[padding].any().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clock-mode', choices=('off', 'empty', 'on'), default='on')
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local_rank)
    dist.init_process_group('hccl')
    rank = dist.get_rank()
    assert dist.get_world_size() == 64, 'This capture requires exactly 64 ranks'
    assert deep_ep._C.get_npu_arch() == 3510, 'Ascend 950/A5 is required'
    assert deep_ep._C.get_num_ai_cores() >= 32
    root = args.output.resolve()
    output = root / f'rank{rank}'
    output.mkdir(parents=True, exist_ok=False)
    os.environ.pop('EP_DEBUG_CLOCK_DIR', None)
    os.environ.pop('EP_DEBUG_CLOCK_EMPTY', None)
    if args.clock_mode != 'off':
        os.environ['EP_DEBUG_CLOCK_DIR'] = str(root / 'raw')
    if args.clock_mode == 'empty':
        os.environ['EP_DEBUG_CLOCK_EMPTY'] = '1'
    torch.manual_seed(rank)
    idx_weights, idx = torch.topk(torch.rand(4096, 384, device='npu', dtype=torch.float32), 6, sorted=False)
    x = per_token_cast_to_fp8(torch.randn(4096, 7168, device='npu', dtype=torch.bfloat16))
    x = (x[0], x[1].view(torch.int16))
    manifest = dict(status='incomplete', rank=rank, world_size=64, tokens_per_rank=4096,
                    experts=384, topk=6, hidden=7168, dtype='fp8_e4m3fn', scale_layout='row-major',
                    expert_alignment=128, ai_cores=32, aivs=64, cached_handle=False,
                    routing='uniform random top-k', seed=rank, cache='warm; no L2 flush',
                    warmups=10, captured_iteration=11, clock_mode=args.clock_mode,
                    torch=torch.__version__, torch_npu=torch_npu.__version__, deep_ep=deep_ep.__version__,
                    clock_alignment='unverified', ticks_per_us=None)
    manifest['akl_revision'] = 'a697a16953c8d22bc5effc85c0b55b5bf3ba72a1'
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    buffer = deep_ep.EPBuffer(dist.group.WORLD, num_max_tokens_per_rank=4096, hidden=7168,
                              num_topk=6, use_fp8_dispatch=True, explicitly_destroy=True)
    try:
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        for iteration in range(1, 12):
            dist.barrier()
            if iteration == 11:
                start.record()
            result = buffer.dispatch(x=x, topk_idx=idx, topk_weights=idx_weights,
                                     num_experts=384, expert_alignment=128, num_sms=32,
                                     do_expand=True, do_cpu_sync=True, do_zero_padding=True,
                                     async_with_compute_stream=True)
            result[-1].current_stream_wait()
            if iteration == 11:
                end.record()
            torch.npu.synchronize()
        sample_us = start.elapsed_time(end) * 1000
        deep_ep._C.export_dispatch_debug_clock()
        check_output(x, idx, idx_weights, result)
        if args.clock_mode != 'off':
            folders = sorted((root / 'raw').glob(f'*/rank{rank}-pid*-launch*'))
            assert {p.parent.name for p in folders} == {'dispatch', 'dispatch_copy_epilogue'}
            assert len(folders) == 2
            sources = list((Path(deep_ep.__file__).parent / 'include/deep_ep').rglob('dispatch*.hpp'))
            if args.clock_mode == 'on':
                from akl.semantic import decode_capture, event_map
                mapping = event_map(sources)
            for folder in folders:
                meta = json.loads((folder / 'capture.json').read_text())
                assert meta['rank'] == rank and meta['blocks'] == 64 and meta['recorder_capacity'] == 32
                if args.clock_mode == 'on':
                    _, events, warnings = decode_capture(folder, mapping)
                    assert not warnings and {e['block'] for e in events} == set(range(64))
                else:
                    words = 8 + 2 * meta['capacity']
                    raw = (folder / 'trace.bin').read_bytes()
                    assert len(raw) == 64 * words * 8
                    rows = struct.iter_unpack(f'<{words}Q', raw)
                    assert all(r[:2] == (0x414B4C5452433031, 1) and r[2:4] == (0, 0) and r[4] == b and r[7] == 1
                               for b, r in enumerate(rows))
        manifest.update(status='passed', precision='bitwise FP8/scales/weights/padding', sample_us=sample_us,
                        timing='NPU event; trace modes include diagnostic allocation and flush overhead')
        manifest_path.write_text(json.dumps(manifest, indent=2))
        dist.barrier()
        print(f'rank={rank} mode={args.clock_mode} iteration=11 sample_us={sample_us:.3f} precision=PASS', flush=True)
    finally:
        try:
            deep_ep._C.export_dispatch_debug_clock()
        finally:
            buffer.destroy()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
