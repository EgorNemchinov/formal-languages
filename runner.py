import time
import argparse
import numpy as np
from collections import defaultdict

import matrix_operations
from matrix_convert import to_type, from_type, to_gpu, from_gpu
from matrix_operations import update_matrix
from parsing import parse_grammar, parse_graph_get_matrices, products_set

from logger import log
import logger

gpu = False


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
    matrices = from_type(matrices)

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
    log('Setting up nonterm-to-product lookup dictionary')
    inverse_by_nonterm = defaultdict(set)
    for body, heads in inverse_grammar.items():
        assert type(body) is tuple, 'Left terminals in grammar: {}'.format(body)
        for head in heads:
            if body[0] != head:
                inverse_by_nonterm[body[0]].add((head, body))
            if body[1] != head:
                inverse_by_nonterm[body[1]].add((head, body))
    log('Start solving')

    counter = 0
    to_recalculate = set(products_set(grammar))
    log('Initial products: {}'.format(to_recalculate))
    while to_recalculate:
        counter += 1
        head, body = to_recalculate.pop()
        assert type(body) is tuple, 'Body is either str or tuple, not {}'.format(type(body))
        is_changed = update_matrix(matrices, head, body)
        if not is_changed:
            continue
        to_recalculate |= inverse_by_nonterm[head]
    log('Finished solving, processed {} products'.format(counter))


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
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Save output into file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs into console')
    parser.add_argument('-c', '--cpu', action='store_true', help='Run on CPU simple bool multiplication')
    parser.add_argument('-t', '--type', type=str, default='bool', help='Compress bools to type')
    parser.add_argument('-s', '--shared_memory', action='store_true', help='Use multiplication with shared memory')
    parser.add_argument('-r', '--run_time', action='store_true', help='Print solution and run times')
    args = parser.parse_args()

    logger.verbose = args.verbose
    gpu = not args.cpu
    matrix_operations.gpu = gpu
    matrix_operations.shared_memory = args.shared_memory

    time_dict = {}
    solution = run(args.grammar_path, args.graph_path, args.type, time_dict=time_dict)

    if args.output_path is None:
        print(solution)
    else:
        with open(args.output_path, 'w') as f:
            f.write(solution)

    run_time_string = 'Solving took {}s'.format(time_dict['solution_time'])
    total_time_string = 'Total run time is {}s'.format(time_dict['run_time'])
    log(run_time_string)
    log(total_time_string)

    if not args.verbose and args.run_time:
        print(run_time_string)
        print(total_time_string)
