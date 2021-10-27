import itertools

from imgaug.augmenters import Augmenter, Sequential
from imgaug.augmentables.batches import _AUGMENTABLE_NAMES
from imgaug.multicore import Pool
import imgaug as ia

from bpyutils import parallel
from bpyutils.util.types import build_fn

from deeply.const import N_JOBS

def _augment_batch(sequence, batch, parents = None, hooks = None):
    return sequence.augment_batches_(batch, parents = parents, hooks = hooks)

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
                        self.combinations.append(Sequential(subset))
                        
            else:
                raise ValueError("Expected None or Augmenter or list of Augmenter, "
                    "got %s." % (type(children),))

    def _augment_batch_(self, batch, random_state, parents, hooks):
        batch_copy   = batch.deepcopy()
        batch_result = batch_copy

        with batch.propagation_hooks_ctx(self, hooks, parents):
            # with parallel.no_daemon_pool(processes = N_JOBS) as pool:
            #     function_   = build_fn(_augment_batch, batch = batch, parents = parents + [self], hooks = hooks)
            #     results     = pool.imap(function_, self.combinations)

            #     for result in results:
            #         batch_result = self._append_batch(batch_result, result)
            for sequential in self.combinations:
                result = sequential.augment_batch_(batch,
                    parents = parents + [self],
                    hooks   = hooks
                )

                batch_result = self._append_batch(batch_result, result)

                result = batch_copy.deepcopy()

        batch_result = self._drop_batch_duplicates(batch_result)

        return batch_result

    def _drop_batch_duplicates(self, batch):
        # TODO: Implement...
        return batch

    def _append_batch(self, batch, value):
        for augmentable_name in _AUGMENTABLE_NAMES:
            prev = getattr(batch, augmentable_name, None)
            next = getattr(value, augmentable_name, None)

            if next is not None:
                value = prev

                if prev is not None:
                    value = list(itertools.chain(prev, next))
                else:
                    value = next

                setattr(batch, augmentable_name, value)
                
        return batch

    def get_parameters(self):
        return []