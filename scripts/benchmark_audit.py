#!/usr/bin/env python3
"""Offline reproducible scope/centroid benchmark; never opens the production DB."""
import argparse
import json
import math
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from store import ContextStore
from index import EmbeddingIndex
from segmenter import SessionSegmenter


def measurements(call, samples):
    values = []
    for _ in range(samples):
        started = time.perf_counter()
        call()
        values.append(round((time.perf_counter() - started) * 1000, 3))
    ordered = sorted(values)
    return {'samples_ms': values, 'p50_ms': statistics.median(values),
            'sample_p95_ms': ordered[math.ceil(.95 * len(ordered)) - 1],
            'sample_p99_ms': ordered[math.ceil(.99 * len(ordered)) - 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=100_000)
    parser.add_argument('--dimension', type=int, default=1024)
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mneme-benchmark-') as directory:
        path = str(Path(directory) / 'benchmark.db')
        store = ContextStore(path, embedding_dim=args.dimension)
        config = {'embedding_model': 'benchmark', 'embedding_dim': args.dimension,
                  'centroid_window': 200, 'centroid_cache_size': 100}
        index = EmbeddingIndex(path, config=config)
        vector = [1.] + [0.] * (args.dimension - 1)
        blob = struct.pack(f'{args.dimension}f', *vector)
        conn = store._get_connection()
        started = time.perf_counter()
        for start in range(0, args.rows, 1000):
            events, embeddings = [], []
            for number in range(start, min(start + 1000, args.rows)):
                sid = f'S{number // 1000}'
                eid = f'{number:010d}'
                events.append((eid, sid, 'seg_' + sid + '_1', eid, 'user_message', 'user', 'benchmark'))
                embeddings.append((eid, 'seg_' + sid + '_1', blob, 'benchmark'))
            conn.executemany('INSERT INTO events(id,session_id,segment_id,timestamp,type,role,content) VALUES(?,?,?,?,?,?,?)', events)
            conn.executemany('INSERT INTO embedding_index(event_id,segment_id,embedding,embedding_model_id) VALUES(?,?,?,?)', embeddings)
        conn.commit()
        setup_seconds = time.perf_counter() - started
        conn.close()
        segmenter = SessionSegmenter(store, index, 'seg_S0_1', config=config)
        segmenter.bind_session('S0', 'seg_S0_1')
        report = {'rows': args.rows, 'dimension': args.dimension,
                  'setup_seconds': round(setup_seconds, 3), 'vec_available': index._vec_available,
                  'scope_candidates': min(1000, args.rows)}
        report['scoped_top5'] = measurements(lambda: index._search_vec0(vector, 'S0', top_k=5), args.samples)
        report['global_top5'] = measurements(lambda: index._search_vec0_multi(vector, None, None, 5), args.samples)
        report['centroid_first'] = measurements(lambda: segmenter.get_centroid('seg_S0_1'), 1)
        report['centroid_cached'] = measurements(lambda: segmenter.get_centroid('seg_S0_1'), args.samples)
        segmenter.close()
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
