import time
import argparse
import numpy as np
from matrix_operations import to_type, from_type, to_gpu, from_gpu

from parsing import parse_grammar, parse_graph_get_matrices

verbose, gpu, shared_memory = False, False, False


def log(message):
    if verbose:
        print('-> ' + message)


def run(grammar_path, graph_path, type='bool', time_dict=None):
    run_begin_time = time.time()
    grammar, inverse_grammar = parse_grammar(grammar_path)
    matrices = parse_graph_get_matrices(graph_path, grammar, inverse_grammar)
    process_grammar(grammar, inverse_grammar)

    matrices = to_type(matrices, type)
    if gpu:
        matrices = to_gpu(matrices)

    solution_begin_time = time.time()
    solve(grammar, inverse_grammar, matrices)
    solution_end_time = time.time()

    if gpu:
        matrices = from_gpu(matrices)

    answer = solution_string(matrices)
    run_end_time = time.time()

    if time_dict is not None:
        time_dict['solution_time'] = solution_end_time - solution_begin_time
        time_dict['run_time'] = run_end_time - run_begin_time
    return answer


def calculate_matrices(grammar, inv_grammar, graph, graph_size):
    matrices = {i: np.zeros((graph_size, graph_size), dtype=np.bool) for i in grammar}
    for row, verts in graph.items():
        for col, value in verts.items():
            if value in inv_grammar:
                for nonterminal in inv_grammar[value]:
                    matrices[nonterminal][row, col] = True
    log('Calculated {} adjacency matrices of shape {}'.format(len(matrices), (graph_size,) * 2))
    return matrices


def process_grammar(grammar, inverse_grammar):
    log('Removing terminals from grammar..')
    terminals = [body for body in inverse_grammar.keys() if type(body) is str]
    for terminal in terminals:
        heads = inverse_grammar.pop(terminal)
        for head in heads:
            grammar[head].remove(terminal)
    log('Successfully removed terminals from grammar. Amount: {}'.format(len(terminals)))


def solve(grammar, inverse_grammar, matrices):
    log('Start solving..')
    log('Finish solving..')


def solution_string(matrices):
    lines = []
    for nonterminal, matrix in matrices.items():
        xs, ys = np.where(matrix)
        pairs = np.vstack((xs, ys)).T + 1
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        pairs_vals = ' '.join(map(lambda pair: ' '.join(pair), pairs.astype('str').tolist()))
        lines.append('{} {}'.format(nonterminal, pairs_vals))
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('grammar_path', type=str, help='File with grammar in CNF')
    parser.add_argument('graph_path', type=str, help='Path to a directional graph')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs into console')
    parser.add_argument('-c', '--cpu', action='store_true', help='Run on CPU simple bool multiplication')
    parser.add_argument('-t', '--type', type=str, default='bool', help='Compress bools to type')
    parser.add_argument('-m', '--memory_shared', action='store_true', help='Use multiplication with shared memory')
    args = parser.parse_args()

    verbose = args.verbose
    gpu = not args.cpu
    shared_memory = args.memory_shared

    time_dict = {}
    print(run(args.grammar_path, args.graph_path, args.type, time_dict=time_dict))
    log('Solving took {} s'.format(time_dict['solution_time']))
    log('Total run time is {} s'.format(time_dict['run_time']))
