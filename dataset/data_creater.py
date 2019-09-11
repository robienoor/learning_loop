import numpy as np
from itertools import chain, combinations, product

# want data that is 1000 reviews with 3 args
# want to try out different distributions of positive, neutral and negatives

positive_args = ['a', 'c']
negative_args = ['b']

pos_attacks = list(product(positive_args, negative_args))
neg_attacks = list(product(negative_args, positive_args))

all_args = positive_args + negative_args
all_attacks = neg_attacks + pos_attacks
powerset = list(chain.from_iterable(combinations(all_attacks, r) for r in range(len(all_args)+2)))

print(powerset)
print(len(powerset))