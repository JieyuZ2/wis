from ..dataset import TextDataset, NGramLFGenerator


def generate_lfs(data: TextDataset, n_lfs: int):
    g = NGramLFGenerator(data.raw_data, data.y, target_labels=[1], ngram_range=(1, 2), min_acc_gain=0.1, min_support=0.1)
    applier = g.generate(n_lfs=n_lfs, mode='accurate')
    # applier = g.generate(n_lfs=n_lfs, mode='random')
    return applier


def apply_lfs(model, data: TextDataset):
    L = model.apply(data.raw_data)
    return L
