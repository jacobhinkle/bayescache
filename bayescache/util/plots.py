import numpy as np
import seaborn as sns


def plot_loss_distribution(losses):
    """Plot aggregate results of series.

    Parameters
    ----------
    losses : pd.DataFrame
      Dataframe containing random walks as columns.
    """
    results = reward_stats(losses)
    sns.set_context("paper")
    sns.tsplot(results)


def loss_stats(losses):
    """Get mean and standard deviation of loss curves across seeds.

    Parameters
    ----------
    losses : pd.DataFrame
      Dataframe containing loss curves across seeds as columns.
    """
    mean = np.array(losses.mean(axis=1))
    std_dev = np.array(losses.std(axis=1))
    upper = mean + std_dev
    lower = mean - std_dev
    stats = [upper, mean, lower]
    return stats
