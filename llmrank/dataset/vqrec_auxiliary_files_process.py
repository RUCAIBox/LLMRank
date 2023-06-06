import os
import argparse
import numpy as np
import faiss


def parse_args():
    # Basic
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--input_path', type=str, default='dataset')
    parser.add_argument('--output_path', type=str, default='dataset')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--suffix', type=str, default='feat1CLS')
    parser.add_argument('--plm_size', type=int, default=768)

    # PQ
    parser.add_argument("--subvector_num", type=int, default=32, help='16/24/32/48/64/96')
    parser.add_argument("--n_centroid", type=int, default=8)
    parser.add_argument("--use_gpu", type=int, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    feat_path = os.path.join(args.input_path, args.dataset, f'{args.dataset}.{args.suffix}')
    loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, args.plm_size)
    print(f'Load {loaded_feat.shape} from {feat_path}.')

    save_index_path = os.path.join(
        args.output_path,
        args.dataset,
        f"{args.dataset}.OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x{args.n_centroid}.index")

    if args.use_gpu:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024 * 1024 * 512)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = args.subvector_num >= 56
    faiss.omp_set_num_threads(32)

    index = faiss.index_factory(args.plm_size,
        f"OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x{args.n_centroid}", faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    if args.use_gpu:
        index = faiss.index_cpu_to_gpu(res, args.gpu_id, index, co)
    index.train(loaded_feat)
    index.add(loaded_feat)
    if args.use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, save_index_path)
