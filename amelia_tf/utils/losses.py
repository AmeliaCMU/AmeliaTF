import torch

from torch.nn import functional as F
from amelia_tf.utils.utils import separate_ego_agent

def marginal_loss(
    pred_scores: torch.tensor, mu: torch.tensor, sigma: torch.tensor, target: torch.tensor,
    ego_agent: torch.tensor = None, agent_mask: torch.tensor = None, epoch: int = 0,
    max_epochs: int = 100, add_diversity: bool = True
) -> torch.tensor:
    """ Computes a classification and regression loss for each agent independently. We select the
    future with minimum loss for each agent separately, and back-propagate the error correspondingly.
    We follow these references:
        - MTR: https://arxiv.org/pdf/2209.13508.pdf
        - SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf

    Inputs:
    -------
        pred_scores[torch.tensor(B, A, T, N)]: tensor containing the predictions scores.
            B: batch size
            A: number of agents
            T: trajectory length
            N: number of predicted heads
        mu[torch.tensor(B, A, T, N, D)]: tensor containing the prediction means.
            D: number of dimensions
        sigma[torch.tensor(B, A, T, N, D)]: tensor containing the prediction standard deviations.
        target[torch.tensor(B, A, T, D)]: tensor containing the ground truth trajectory.

    Output
    ------
        error[torch.tensor]: scalar value representing the marginal loss.

    """
    B, A, T, N, D = mu.size()

    if not ego_agent is None:
        A = 1
        assert ego_agent.shape[0] == mu.shape[0]
        ego_agent = ego_agent
        # Slice ego agent for predictions
        mu = separate_ego_agent(mu, ego_agent)
        # Slice ego agent for sigmas
        sigma = separate_ego_agent(sigma, ego_agent)
        # Slice ego agent for scores
        pred_scores = separate_ego_agent(pred_scores, ego_agent)
        # Slice ego agent for target
        target = separate_ego_agent(target, ego_agent)
        # Slice ego agent for masks
        agent_mask = None if agent_mask is None else separate_ego_agent(agent_mask, ego_agent)

    # distance: (B, A, T, N, D) -> (B, A, T, N)
    distance = (mu - target[...,None,:]).norm(dim=-1)
    if agent_mask is None:
        # agg_distance: (B, A, T, N) -> (B, A, N)
        agg_distance = distance.mean(dim=2)

        # index of independent future with smallest error
        # gt_idx: (B, A)
        gt_idx = agg_distance.argmin(dim=-1)

        # select the correct independent future; mask all else
        mask = F.one_hot(gt_idx, num_classes = N)[..., None, :, None].repeat(1, 1, T, 1, D)
        mu = mu * mask
        sigma = sigma * mask
        target = target[..., None, :].repeat(1, 1, 1, N, 1) * mask

        loss_cls = F.cross_entropy(
            input=pred_scores.flatten(0, 1), target=gt_idx.flatten(), reduction='mean', ignore_index=N)
        loss_reg = F.gaussian_nll_loss(mu, target, sigma)
    else:
        agent_mask = agent_mask.view(-1, T)
        distance = distance.view(-1, T, N)
        BA, _, _ = distance.shape
        agg_distance = torch.zeros(BA, N).to(agent_mask.device)
        for ba in range(BA):
            amask = agent_mask[ba]
            agg_distance[ba] = distance[ba, amask].mean(dim=0)
        agg_distance = agg_distance.view(B, A, N)

        # index of independent future with smallest error
        # gt_idx: (B, A)
        gt_idx = agg_distance.argmin(dim=-1)

        # TODO: debug
        # Select the correct independent future; mask all else
        amask = agent_mask.view(B, A, T, 1, 1).repeat(1, 1, 1, N, D)
        mask = F.one_hot(gt_idx, num_classes = N)[..., None, :, None].repeat(1, 1, T, 1, D) * amask
        mu = mu * mask
        sigma = sigma * mask
        target = target[..., None, :].repeat(1, 1, 1, N, 1) * mask

    loss_cls = F.cross_entropy(
        input=pred_scores.flatten(0, 1), target=gt_idx.flatten(), reduction='mean', ignore_index=N)
    loss_reg = F.gaussian_nll_loss(mu,target,sigma)

    return loss_cls + loss_reg

def joint_loss(
    pred_scores: torch.tensor, mu: torch.tensor, sigma: torch.tensor, target: torch.tensor,
    epoch: int = 0, max_epochs: int = 100, add_diversity: bool = True, ego_agent = None
) -> torch.tensor:
    """ Computes a classification and regression loss for the scene jointly. We treat each future to
    be coherent futures across all agents. Thus, we aggregate the loss across all agents and time
    steps. We only back-propagate the loss through the individual future that most closely matches
    the ground-truth in terms of displacement loss.
    We follow these references:
        - MTR: https://arxiv.org/pdf/2209.13508.pdf
        - SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf

    Inputs:
    -------
        pred_scores[torch.tensor(B, A, T, N)]: tensor containing the predictions scores.
            B: batch size
            A: number of agents
            T: trajectory length
            N: number of predicted heads
        mu[torch.tensor(B, A, T, N, D)]: tensor containing the prediction means.
            D: number of dimensions
        sigma[torch.tensor(B, A, T, N, D)]: tensor containing the prediction standard deviations.
        target[torch.tensor(B, A, T, D)]: tensor containing the ground truth trajectory.

    Output
    ------
        error[torch.tensor]: scalar value representing the joint loss.

    """
    B, A, T, N, D = mu.size()

    # distance: (B, A, T, N, D) -> (B, A, T, N)
    distance = (mu - target[...,None,:]).norm(dim=-1)
    # agg_distance: (B, A, T, N) -> (B, A, N) -> (B, N)
    agg_distance = distance.mean(dim=(2, 1))

    # index of joint future with smallest error
    # gt_idx: (B)
    gt_idx = agg_distance.argmin(dim=-1)

    # select the correct joint future; mask all else
    mask = F.one_hot(gt_idx, num_classes = N)[:, None, None, :, None].repeat(1, A, T, 1, D)
    mu = mu * mask
    sigma = sigma * mask
    target = target[..., None, :].repeat(1, 1, 1, N, 1) * mask

    loss_cls = F.cross_entropy(
        input=pred_scores.flatten(0, 1), target=gt_idx[:, None].repeat(1, A).flatten(),
        reduction='mean', ignore_index=N)
    loss_reg = F.gaussian_nll_loss(mu, target, sigma)

    return loss_cls + loss_reg

def diversity_loss(pred: torch.tensor, sigma_d: float = 0.001) -> torch.tensor:
    B, A, T, N, D = pred.shape
    # ----------------------
    # TODO: vectorize
    # ----------------------
    diversity_loss = 0.0
    for n1 in range(N):
        for n2 in range(N):
            if n1 == n2:
                continue

            y_n1 = pred[:, :, :, n1]
            y_n2 = pred[:, :, :, n2]
            diversity_loss += torch.exp(-(y_n1 - y_n2).norm(dim=1) / sigma_d).mean(dim=(2, 1))

    return (diversity_loss / (N * (N - 1))).sum()


def lmbd_marginal_joint_loss(
    pred_scores: torch.tensor, mu: torch.tensor, sigma: torch.tensor, target: torch.tensor,
    lmbd: float = 0.5, epoch: int = 0, max_epochs: int = 100, add_diversity: bool = True, ego_agent = None
) -> torch.tensor:
    """ Computes a classification and regression loss for the scene marginally and jointly. We treat
    each future to be coherent futures across all agents. Thus, we aggregate the loss across all agents
    and time steps. However, we also consider each element independently.
    We follow these references:
        - MTR: https://arxiv.org/pdf/2209.13508.pdf
        - SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf

    Inputs:
    -------
        pred_scores[torch.tensor(B, A, T, N)]: tensor containing the predictions scores.
            B: batch size
            A: number of agents
            T: trajectory length
            N: number of predicted heads
        mu[torch.tensor(B, A, T, N, D)]: tensor containing the prediction means.
            D: number of dimensions
        sigma[torch.tensor(B, A, T, N, D)]: tensor containing the prediction standard deviations.
        target[torch.tensor(B, A, T, D)]: tensor containing the ground truth trajectory.

    Output
    ------
        error[torch.tensor]: scalar value representing the joint loss.

    """
    assert lmbd >= 0 and lmbd < 1.0
    m = marginal_loss(pred_scores, mu, sigma, target)
    j = joint_loss(pred_scores, mu, sigma, target)

    if add_diversity:
        d = diversity_loss(mu)
        return (1.0 - lmbd) * m + lmbd * j + 0.1 * d

    return lmbd * m + (1 - lmbd) * j

def weighted_marginal_joint_loss(
    pred_scores: torch.tensor, mu: torch.tensor, sigma: torch.tensor, target: torch.tensor,
    lmbd: float = 0.5, epoch: int = 0, max_epochs: int = 100, add_diversity: bool = True
) -> torch.tensor:
    """ Computes a classification and regression loss for the scene marginally and jointly. We treat
    each future to be coherent futures across all agents. Thus, we aggregate the loss across all agents
    and time steps. However, we also consider each element independently.
    We follow these references:
        - MTR: https://arxiv.org/pdf/2209.13508.pdf
        - SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf

    Inputs:
    -------
        pred_scores[torch.tensor(B, A, T, N)]: tensor containing the predictions scores.
            B: batch size
            A: number of agents
            T: trajectory length
            N: number of predicted heads
        mu[torch.tensor(B, A, T, N, D)]: tensor containing the prediction means.
            D: number of dimensions
        sigma[torch.tensor(B, A, T, N, D)]: tensor containing the prediction standard deviations.
        target[torch.tensor(B, A, T, D)]: tensor containing the ground truth trajectory.

    Output
    ------
        error[torch.tensor]: scalar value representing the joint loss.

    """
    assert epoch > 0
    w = epoch / max_epochs

    m = marginal_loss(pred_scores, mu, sigma, target)
    j = joint_loss(pred_scores, mu, sigma, target)

    if add_diversity:
        d = diversity_loss(mu)
        return (1.0 - w) * m + w * j + 0.1 * d

    return (1.0 - w) * m + w * j