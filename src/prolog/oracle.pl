:- use_module(library(http/json)).
:- use_module(library(lists)).

% This file intentionally contains no task answers. The Python harness writes a
% task-specific temporary KB and invokes only allow-listed predicates/arity.

print_json(Term) :-
    atom_json_term(Atom, Term, []),
    writeln(Atom).
