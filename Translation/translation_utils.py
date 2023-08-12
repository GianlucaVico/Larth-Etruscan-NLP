"""
Common methods for the translation tasks.
"""
from typing import List, Dict, Tuple, Any
from matplotlib import pyplot as plt
from itertools import product
import sacrebleu


def compute_metrics(sys: List[str], ref: List[str]) -> Dict[str, float]:
    """
    Compute BLEU, chr-F and TER

    Args:
        sys: list of predicted translations
        ref: list of reference translations

    Returns:
        Dictionary with the metrics. Keys: ["bleu", "chrf", "ter"]
    """
    bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True).score
    chrf = sacrebleu.corpus_chrf(sys, [ref]).score
    ter = sacrebleu.corpus_ter(
        sys, [ref], normalized=True, no_punct=True, case_sensitive=False
    ).score

    return {"bleu": bleu, "chrf": chrf, "ter": ter}


def plot_results(
    results: Dict[str, List[float]], title: str, figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot the metrics' distributions

    Args:
        results: dictionary with the distribution of each metrics
        title: title of the plot
        figsize: size of the figure
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for i, j in enumerate(["bleu", "chrf", "ter"]):
        axs[i].boxplot(results[j])
        axs[i].set_xticklabels([j])
        # axs[i].set_ylabel("Score")
        axs[i].grid()
    fig.suptitle(title)
    plt.show()


def print_example(example: Tuple[List[str], List[str], List[str]]) -> None:
    """
    Print some examples

    Args:
        example: tuple with the list of Etruscan texts, the list of
            reference translations and the list of predicted translations
    """
    n = len(example[0])
    for i in range(n):
        print("-" * 60)
        print("Etruscan:", example[0][i])
        print("Reference:", example[1][i])
        print("Prediction:", example[2][i])
        print("-" * 60)


def generate_parameters(parameter_space: Dict[str, List[Any]]):
    """
    Generate the paramters given the list of possible parameter values

    Args:
        parameter_space: dictionary with the values of each parameter
    """
    # From ParameterGrid in sklearn.model_selection
    items = sorted(parameter_space.items())
    keys, values = zip(*items)
    for v in product(*values):
        yield dict(zip(keys, v))
