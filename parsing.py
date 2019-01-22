from collections import defaultdict
import numpy as np


def parse_grammar(file_path):
    grammar, inverse_grammar = defaultdict(set), defaultdict(set)
    with open(file_path, 'rt') as grammar_file:
        for line in grammar_file:
            terms = line.split()
            assert len(terms) in (2, 3), 'Wrong grammar input format. Got {} terms'.format(len(terms))
            if len(terms) == 2:
                grammar[terms[0]].add(terms[1])
                inverse_grammar[terms[1]].add(terms[0])
            elif len(terms) == 3:
                grammar[terms[0]].add((terms[1], terms[2]))
                inverse_grammar[terms[1], terms[2]].add(terms[0])

    return grammar, inverse_grammar


def parse_graph_get_matrices(file_path, grammar, inv_grammar):
    result_graph = defaultdict(dict)
    max_node = 0
    with open(file_path, 'rt') as graph_file:
        for line in graph_file:
            terms = line.split()
            assert len(terms) == 3, 'Wrong graph input format. Expected 3 terms, got {}'.format(len(terms))
            from_vert, to_vert = int(terms[0]), int(terms[2].rstrip(','))
            # assert from_vert > 0 and to_vert > 0, 'Vertices indices must be above zero'
            max_node = max(max_node, from_vert, to_vert)
            result_graph[from_vert][to_vert] = terms[1]
    max_node += 1
    matrices = {i: np.zeros((max_node, max_node), dtype='bool') for i in grammar}
    for row, verts in result_graph.items():
        for col, value in verts.items():
            if value in inv_grammar:
                for nonterminal in inv_grammar[value]:
                    matrices[nonterminal][row, col] = True
    return matrices


def products_set(grammar):
    products = set()
    for head in grammar:
        for body in grammar[head]:
            products.add((head, body))
    return products
