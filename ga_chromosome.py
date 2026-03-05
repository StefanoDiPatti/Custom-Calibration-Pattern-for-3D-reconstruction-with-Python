import cv2
import numpy as np
from random import randrange, randint, random
from ga_constants import GAConstants


class GAChromosome:
    FLAGS = [
        cv2.CALIB_FIX_ASPECT_RATIO,
        cv2.CALIB_ZERO_TANGENT_DIST,
        cv2.CALIB_USE_INTRINSIC_GUESS,
        cv2.CALIB_RATIONAL_MODEL,
        cv2.CALIB_FIX_K3,
        cv2.CALIB_FIX_K4,
        cv2.CALIB_FIX_K5,
    ]
    FLAGS_STR = [
        'FIX_ASPECT_RATIO', 'ZERO_TANGENT_DIST', 'USE_INTRINSIC_GUESS',
        'RATIONAL_MODEL', 'FIX_K3', 'FIX_K4', 'FIX_K5'
    ]
    N_FLAGS = len(FLAGS)

    def __init__(self, n_images: int, flags: list = None, genes: list = None):
        self.n_images = n_images
        if flags is not None and genes is not None:
            assert len(flags) == self.N_FLAGS
            assert len(genes) == n_images
            self.flags = flags
            self.genes = genes
        else:
            done = False
            while not done:
                self.flags = [
                    (True if randint(0, 1) > 0 else False) if GAConstants.OPTIMIZE_FLAGS else False
                    for _ in range(self.N_FLAGS)
                ]
                self.genes = [True if randint(0, 1) > 0 else False for _ in range(n_images)]
                if self.is_valid():
                    done = True

    def is_valid(self) -> bool:
        return self.genes.count(True) >= GAConstants.MIN_GENES

    def get_flags(self) -> int:
        result = 0
        for i, flag in enumerate(self.FLAGS):
            if self.flags[i]:
                result |= flag
        return result

    def active_indices(self) -> list:
        return [i for i, g in enumerate(self.genes) if g]

    def __str__(self) -> str:
        flag_names = [self.FLAGS_STR[i] for i in range(self.N_FLAGS) if self.flags[i]]
        active = self.active_indices()
        return f"Flags: {flag_names} | Images: {active}"

    @staticmethod
    def to_string(c) -> str:
        return ''.join('1' if f else '0' for f in c.flags) + \
               ''.join('1' if g else '0' for g in c.genes)

    @staticmethod
    def clone(c):
        return GAChromosome(
            n_images=c.n_images,
            flags=list(c.flags),
            genes=list(c.genes)
        )

    @staticmethod
    def mutate(c):
        flags = list(c.flags)
        genes = list(c.genes)
        thr = 0.25
        p = random()
        if GAConstants.OPTIMIZE_FLAGS and p < thr:
            idx = randrange(GAChromosome.N_FLAGS)
            flags[idx] = not flags[idx]
        else:
            idx = randrange(c.n_images)
            genes[idx] = not genes[idx]
        return GAChromosome(c.n_images, flags=flags, genes=genes)

    @staticmethod
    def crossover(c1, c2):
        p = randrange(c1.n_images)
        genes1 = c1.genes[:p] + c2.genes[p:]
        genes2 = c2.genes[:p] + c1.genes[p:]
        return (GAChromosome(c1.n_images, flags=list(c1.flags), genes=genes1),
                GAChromosome(c2.n_images, flags=list(c2.flags), genes=genes2))
