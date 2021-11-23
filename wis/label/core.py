ABSTAIN = -1

EXCLUSIVE = 0
OVERLAP = 1
INCLUSION = 2
INCLUDED = 3
EQUAL = 4
UNKNOWN = -10
# SYMMETRIC = [OVERLAP, EXCLUSIVE, EQUAL]
SYMMETRIC = [UNKNOWN, OVERLAP, EXCLUSIVE, EQUAL]
# RELATIONS = [OVERLAP, EXCLUSIVE, EQUAL, INCLUSION]
RELATIONS = [UNKNOWN, OVERLAP, EXCLUSIVE, EQUAL, INCLUSION, INCLUDED]

MAP_STR_TO_VALUE = {
    'equal'    : EQUAL,
    'exclusive': EXCLUSIVE,
    'overlap'  : OVERLAP,
    'unknown'  : UNKNOWN,
    'inclusion': INCLUSION,
    'included' : INCLUDED
}

MAP_VALUE_TO_STR = {
    EQUAL    : 'equal',
    EXCLUSIVE: 'exclusive',
    OVERLAP  : 'overlap',
    UNKNOWN  : 'unknown',
    INCLUSION: 'inclusion',
    INCLUDED : 'included'
}


def relation_value(v):
    if isinstance(v, str):
        return MAP_STR_TO_VALUE[v]
    elif v in RELATIONS:
        return v
    else:
        raise ValueError(f'Unrecognized relation value {v}')


def reverse_relation(r):
    if r in SYMMETRIC:
        return r
    else:
        if r == INCLUSION:
            return INCLUDED
        elif r == INCLUDED:
            return INCLUSION
