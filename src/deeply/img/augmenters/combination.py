import itertools

from bpyutils.util.array import squash

from imgaug.augmenters import Augmenter, Sequential
from imgaug.augmentables.batches import _AUGMENTABLE_NAMES
import imgaug as ia

class Combination(Augmenter):
    def __init__(self, children = None, *args, **kwargs):
        self._super = super(Combination, self)
        self._super.__init__(*args, **kwargs)
        
        self.combinations = []
        
        if children is not None:
            if isinstance(children, Augmenter):
                self.combinations = [children]
            if ia.is_iterable(children):
                assert all([isinstance(child, Augmenter) for child in children]), \
                    ("Expected all children to be augmenters, got types %s." % 
                        (", ".join([str(type(v)) for v in children])))

                length = len(children)

                for L in range(1, length + 1):
                    for subset in itertools.combinations(children, L):
                        subset = squash(subset)
                        self.combinations.append(subset)
                        
            else:
                raise ValueError("Expected None or Augmenter or list of Augmenter, "
                    "got %s." % (type(children),))

    def _augment_batch_(self, batch, random_state, parents, hooks):
        batch_copy   = batch.deepcopy()
        batch_result = batch.deepcopy()

        with batch.propagation_hooks_ctx(self, hooks, parents):
            for combination in self.combinations:
                sequential = Sequential(combination)
                result = sequential.augment_batch_(batch,
                    parents  = parents + [self],
                    hooks    = hooks
                )

                batch_result = self._append_batch(batch_result, result)

                batch  = batch_copy.deepcopy()

        batch_result = self._drop_batch_duplicates(batch_result)

        return batch_result

    def _drop_batch_duplicates(self, batch):
        # TODO: Implement...
        return batch

    def _append_batch(self, batch, value):
        for augmentable_name in _AUGMENTABLE_NAMES:
            prev = getattr(batch, augmentable_name, None)
            next = getattr(value, augmentable_name, None)

            if next:
                value = prev

                if prev:
                    value = list(itertools.chain(prev, next)) 
                else:
                    value = next

                setattr(batch, augmentable_name, value)
                
        return batch

    def get_parameters(self):
        return []