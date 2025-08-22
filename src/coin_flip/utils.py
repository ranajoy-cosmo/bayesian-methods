import numpy as np
import plotly.express as px

def flip_coin(n_flips=100, n_ensembles=1, p_head=0.5):
    """
    Simulates flipping a coin multiple times and returns the outcomes.
    This function flips a coin `n_flips` times for `n_ensembles` ensembles, with a probability `p_head` of landing heads.
    If n_ensemble is set to 1, it will return a single array of outcomes.

    Args:
        n_flips (int, optional): Number of flips of the coin per ensemble. Defaults to 100.
        n_ensembles (int, optional): Number of ensembles to simulate. Defaults to 1.
        p_head (float, optional): Probability of heads in each flip. Defaults to 0.5.
    Returns:
        np.ndarray: A 2D array of outcomes where 1 represents heads and 0 represents tails. Each row corresponds to a single ensemble of flips.
        if `n_ensembles` is 1, it returns a 1D array of outcomes.
    """

    head_outcomes = np.random.choice(
        np.array([1, 0]), 
        size=n_flips * n_ensembles, 
        p=[p_head, 1 - p_head]).reshape(n_ensembles, n_flips)
    
    if n_ensembles == 1:
        return head_outcomes[0]
    else:
        return head_outcomes
    
def is_coin_fair(head_outcomes, sigma_threshold=2.0):
    """
    Estimates if the coin is fair based on the outcomes of flips.
    A fair coin is defined as one that has an equal probability of landing heads or tails. i.e p_head = 0.5.
    The function uses the maximum likelihood estimate of the probability of heads and checks if it is within a certain number of standard deviations from 0.5.

    Args:
        head_outcomes (np.ndarray): A ID array of coin flip outcomes where 1 represents heads and 0 represents tails.
        sigma_threshold (float, optional): The threshold for the standard deviation to consider the coin fair. Defaults to 2.0. Similar to a 95% confidence interval.
    Returns:
        bool: True if the coin is estimated to be fair, False otherwise.
    """
    n_heads = np.sum(head_outcomes)
    n_flips = len(head_outcomes)
    
    p_head_ML = n_heads / n_flips # The maximum likelihood estimate of p_head
    sigma_p_head_ML = np.sqrt((p_head_ML * (1 - p_head_ML)) / n_flips) # Standard deviation of the maximum likelihood estimate of p_head

    num_sigma_from_fair = abs(p_head_ML - 0.5) / sigma_p_head_ML

    is_fair = num_sigma_from_fair < sigma_threshold

    return is_fair, p_head_ML, sigma_p_head_ML, num_sigma_from_fair

def num_std_away_from_fair(p_head_ML, n_flips):
    """
    Calculates the number of standard deviations away from a fair coin (p_head = 0.5) based on the maximum likelihood estimate of p_head.

    Args:
        p_head_ML (float): The maximum likelihood estimate of the probability of heads.
        n_flips (int): The number of flips.

    Returns:
        float: The number of standard deviations away from a fair coin.
    """
    sigma_p_head_ML = np.sqrt((p_head_ML * (1 - p_head_ML)) / n_flips)
    num_sigma_from_fair = abs(p_head_ML - 0.5) / sigma_p_head_ML
    return num_sigma_from_fair

def num_flips_to_achieve_sigma(sensitivity, sigma_threshold=2.0):
    """
    Calculates the number of flips required to achieve a certain number of standard deviations away from a fair coin.

    Args:
        sigma_threshold (float, optional): The threshold for the number of standard deviations to consider the coin fair. Defaults to 2.0.
        sensitivity (float, optional): The gap between the maximum likelihood estimate of p_head and 0.5.

    Returns:
        int: The number of flips required to achieve the specified number of standard deviations.
    """
    p_head_ML = sensitivity + 0.5  # Assuming sensitivity is the difference from 0.5. Since we square the modulus, this will work for both sides of 0.5.
    n_threshold = p_head_ML * (1 - p_head_ML) * (sigma_threshold ** 2) / (p_head_ML - 0.5) ** 2
    return np.ceil(n_threshold).astype(np.int64)

def create_meshgrid(sens_pars, sigma_pars):
    """
    Create a numpy meshgrid with the sensitivity values on the x-axis and sigma threshold values on the y-axis.

    Args:
        sensitivity_range (np.ndarray): Array of sensitivity values.
        sigma_threshold_range (np.ndarray): Array of sigma threshold values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Meshgrid arrays for sensitivity and sigma threshold.
    """

    # Sensitivity values from 0.01 to 0.5. Exclude 0.5 to avoid division by zero
    sens_range = np.linspace(sens_pars['low'], sens_pars['high'], sens_pars['n'], endpoint=False)
    # Sigma values from 0.5 to 5
    sigma_range = np.linspace(sigma_pars['low'], sigma_pars['high'], sigma_pars['n'])[::-1]
    # Creating the meshgrid
    sens_grid, sigma_grid = np.meshgrid(sens_range, sigma_range)
    return sens_grid, sigma_grid, sens_range, sigma_range

def plot_test_power_heatmap(sens_range, sigma_range, num_flips_grid):
    """
    Plot a heatmap of the number of flips required to achieve a certain sensitivity and sigma threshold.

    Args:
        sens_range (np.ndarray): Array of sensitivity values.
        sigma_range (np.ndarray): Array of sigma threshold values.
        num_flips_grid (np.ndarray): 2D array of number of flips required for each combination of sensitivity and sigma threshold.

    Returns:
        plotly.graph_objects.Figure: The heatmap figure.
    """
    fig = px.imshow(num_flips_grid,
                    labels=dict(x="Sensitivity (gap from 0.5)", y="Sigma Threshold", color="Number of Flips"),
                    x=np.round(sens_range, 2),
                    y=np.round(sigma_range, 2),
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    title="Number of Flips Required to Detect Unfairness with Given Sensitivity and Sigma Threshold")
    fig.update_xaxes(tickangle=45)
    fig.update_layout(coloraxis_colorbar=dict(title="Number of Flips"))
    return fig