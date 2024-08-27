import torch
import numpy as np
from shapely import LineString

def marginal_ade(
    Y_hat: torch.tensor, Y: torch.tensor, mask: torch.tensor = None, scale: int = 1000.0
) -> torch.tensor:
    """ Computes the marginal Average Displacement Error (mADE). It computes the mean error across
    time steps, and selects the future with the smallest error for each agent independently, and
    then computes the mean across the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.

    Output
    ------
        error[torch.tensor]: marginal average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1) # B, A, T, H, 2 -> B, A, T, H
    if mask is None:
        error = error.mean(dim=2)                                  # B, A, T, H    -> B, A, H
    else:
        mask = mask[:, :, -T:]
        error = error.view(B * A, T, -1)
        BA, T, H = error.shape
        mask = mask.view(BA, T)
        error_masked = torch.zeros(BA, H).to(error.device)
        for ba in range(BA):
            amask = mask[ba]
            error_masked[ba] = error[ba, amask].mean(dim=0)
        error = error_masked.view(B, A, H)
    error = error.min(dim=-1)[0]                                   # B, A, H       -> B, A
    return scale * error#.mean()                                   # B, A          -> 1

def marginal_prob_ade(Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor,
                      mask: torch.tensor = None) -> torch.tensor:
    """ Computes the marginal probability-weighted Average Displacement Error (mADE). It computes the
    mean error across time steps and weights it by the scores for the predicted trajectories. Then
    selects the future with the smallest error for each agent independently, and then computes the
    mean across the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.

    Output
    ------
        error[torch.tensor]: marginal average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1) # B, A, T, H, 2 -> B, A, T, H
    if mask is None:
        error = error.mean(dim=2) * Y_hat_scores                   # B, A, T, H    -> B, A, H
    else:
        mask = mask[:, :, -T:]
        error = error.view(B * A, T, -1)
        BA, T, H = error.shape
        mask = mask.view(BA, T)
        error_masked = torch.zeros(BA, H).to(error.device)
        for ba in range(BA):
            amask = mask[ba]
            error_masked[ba] = error[ba, amask].mean(dim=0)
        error = error_masked.view(B, A, H) * Y_hat_scores
    error = error.sum(dim=-1)                                      # B, A, H       -> B, A
    return error#.mean()                                           # B, A          -> 1

def marginal_fde(
    Y_hat: torch.tensor, Y: torch.tensor, mask: torch.tensor = None, scale: int = 1000.0
) -> torch.tensor:
    """ Computes the marginal Final Displacement Error (mFDE). It computes the mean error for the
    last time step, and selects the future with the smallest error for each independently, and then
    computes the mean across the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.

    Output
    ------
        error[torch.tensor]: marginal final displacement error.
    """
    B, A, T, D = Y.size()
    if mask is None:
        error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1) # B, A, H, 2 -> B, A, H
    else:
        mask, Y_hat = mask[:, :, -T:], Y_hat[:, :, -T:]
        # Get the last valid index
        t = (mask != 0).cumsum(-1).argmax(-1)
        x, y = torch.meshgrid(torch.arange(0, B), torch.arange(0, A), indexing='ij')
        Y_T = Y[x, y, t]          # B, A, D
        Y_hat_T = Y_hat[x, y, t]  # B, A, H, D
        error = (Y_hat_T - Y_T[..., None, :]).norm(dim=-1)

    error = error.min(dim=-1)[0]                                      # B, A, H       -> B, A
    return scale * error#.mean()                                      # B, A          -> 1

def marginal_prob_fde(
    Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor, mask: torch.tensor = None
) -> torch.tensor:
    """ Computes the marginal probability-weighted Final Displacement Error (mFDE). It computes the
    mean error for the last time step and weights it by the scores for the predicted trajectories.
    Then, it selects the future with the smallest error for each independently, and then computes the
    mean across the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.

    Output
    ------
        error[torch.tensor]: marginal final displacement error.
    """
    B, A, T, D = Y.size()
    if mask is None:
        error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1) # B, A, H, 2 -> B, A, H
    else:
        # Get the last valid index
        mask, Y_hat = mask[:, :, -T:], Y_hat[:, :, -T:]
        t = (mask != 0).cumsum(-1).argmax(-1)
        x, y = torch.meshgrid(torch.arange(0, B), torch.arange(0, A), indexing='ij')
        Y_hat_T = Y_hat[x, y, t]  # B, A, H, D
        Y_T = Y[x, y, t]          # B, A, D
        error = (Y_hat_T - Y_T[..., None, :]).norm(dim=-1)
    error = error * Y_hat_scores
    error = error.sum(dim=-1)                                         # B, A, H       -> B, A
    return error#.mean()                                              # B, A          -> 1

def joint_ade(Y_hat: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """ Computes the joint Average Displacement Error (jADE). Take the average error over all agents
    within a sample before selecting the best one to use in evaluation, then computes the mean across
    the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.

    Output
    ------
        error[torch.tensor]: joint average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1) # B, A, T, H, 2 -> B, A, T, H

    # error across time and agents
    error = error.mean(dim=(2, 1))                                 # B, A, T, H    -> B, H
    error = error.min(dim=-1)[0]                                   # B, H          -> B
    return error#.mean()                                           # B             -> 1

def joint_prob_ade(Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """ Computes the joint Average Displacement Error (jADE). Take the average error over all agents
    within a sample before selecting the best one to use in evaluation, then computes the mean across
    the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y_hat_scores[torch.tensor(B, A, H)]: scores for the predicted trajectories
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.

    Output
    ------
        error[torch.tensor]: joint average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1) # B, A, T, H, 2 -> B, A, T, H
    error = error * Y_hat_scores[..., None, :]
    # error across time and agents
    error = error.mean(dim=(2, 1))                                 # B, A, T, H    -> B, H
    error = error.sum(dim=-1)                                      # B, H          -> B
    return error#.mean()

def joint_fde(Y_hat: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """ Computes the joint Final Displacement Error (jFDE). Take the final error over all agents
    within a sample before selecting the best one to use in evaluation, then computes the mean across
    the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.

    Output
    ------
        error[torch.tensor]: joint average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1) # B, A, H, 2 -> B, A, H

    # error across agents
    error = error.mean(dim=1)                                         # B, A, H    -> B, H
    error = error.min(dim=-1)[0]                                      # B, H       -> B
    return error#.mean()                                              # B          -> 1

def joint_prob_fde(Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """ Computes the joint Final Displacement Error (jFDE). Take the final error over all agents
    within a sample before selecting the best one to use in evaluation, then computes the mean across
    the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.

    Output
    ------
        error[torch.tensor]: joint average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1) # B, A, H, 2 -> B, A, H
    error = error * Y_hat_scores
    # error across agents
    error = error.mean(dim=1)                                         # B, A, H    -> B, H
    error = error.sum(dim=-1)                                         # B, H       -> B
    return error#.mean()                                              # B          -> 1



# collision implementation
def compute_collision(A, B, coll_thresh: float = 0.3):
    # A: 1, T, D
    # B: N, T, D
    breakpoint()

    coll_sum = 0
    seg_a = np.stack([A[:-1], A[1:]], axis=1)
    seg_a = [LineString(x) for x in seg_a]
    for b_sub in B:
        seg_b = np.stack([b_sub[:-1], b_sub[1:]], axis=1)
        seg_b = [LineString(x) for x in seg_b]
        coll = np.linalg.norm(A - b_sub, axis=-1) <= coll_thresh
        coll[1:] |= [x.intersects(y) for x, y in zip(seg_a, seg_b)]
        breakpoint()
        coll_sum += coll.sum()
    return coll_sum

def compute_collisions_to_gt(
    Y_hat: torch.Tensor, Y_gt: torch.Tensor, num_agents: torch.tensor, ego_agent: torch.tensor,
    coll_thresh: float = 0.3
) -> torch.tensor:
    """ Computes collisions between predicted agent and other agents' ground truth.

    Inputs
    ------
        Y_hat[torch.tensor(B, 1, T, M, D)]: ego agent's prediction.
        Y_gt[torch.tensor(B, A, T, D)]: ground truth scene.
        num_agents[torch.tensor(B)]: number of agents in each scene.
        ego_agent[torch.tensor(B)]: ID of ego agent within the scene.

    Output
    ------
        collisions[torch.tensor(B)]: worst mode's (max. number of) collisions.

    """
    B, A, T, D = Y_gt.shape
    Y_hat = Y_hat[..., -T:, :, :].cpu().numpy()
    Y_gt = Y_gt.cpu().numpy()
    _, _, _, M, _ = Y_hat.shape

    collisions = torch.zeros(size=(B,))

    breakpoint()
    # Iterating over all scenes
    # TODO: add weigh by Y_hat_scores
    for b in range(B):
        # Iterating over all ego-agent modes
        ego_modes = Y_hat[b, 0]                        # T, M, D
        mask = np.zeros(shape=(A,), dtype=bool)        # A
        mask[:num_agents[b]] = True
        mask[ego_agent[b]] = False
        other_Y = Y_gt[b, mask]                        # A-1, T, D
        collisions[b] = max(
            [compute_collision(ego_modes[..., m, :], other_Y, coll_thresh) for m in range(M)])
    return collisions.to('cuda:0')


def compute_collisions_to_pred(Y_hat: torch.Tensor, Y_gt: torch.Tensor, num_agents: torch.tensor,
                               ego_agent:torch.tensor,coll_thresh: float = 0.3) -> torch.tensor:
    """ Computes collisions amongst scene predictions. Assumes one predicted scene per mode.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, M, D)]: scene predictions.
        Y_gt[torch.tensor(B, A, T, D)]: ground truth scene.
        num_agents[torch.tensor(B)]: number of agents in each scene.
        ego_agent[torch.tensor(B)]: ID of ego agent within the scene.

    Output
    ------
        collisions[torch.tensor(B)]: worst mode's (max. number of) collisions.

    """
    B, A, T, D = Y_gt.shape
    Y_hat = Y_hat[..., -T:, :, :].cpu().numpy()
    _, _, _, M, _ = Y_hat.shape

    collisions = torch.zeros(size=(B,))

    # Iterating over all scenes
    for b in range(B):
        ego_modes = Y_hat[b, ego_agent[b]]             # T, M, D
        mask = np.zeros(shape=(A,), dtype=bool)        # A
        mask[:num_agents[b]] = True
        mask[ego_agent[b]] = False
        other_Y = Y_hat[b, mask]                       # A-1, T, M, D
        collisions[b] = max(
            [compute_collision(ego_modes[..., m, :], other_Y[..., m, :], coll_thresh) for m in range(M)])
    return collisions.to('cuda:0')


def compute_collisions_gt2gt(
    Y_hat: torch.Tensor, Y_gt: torch.Tensor, num_agents: torch.tensor, ego_agent: torch.tensor,
    coll_thresh: float = 0.3
) -> torch.tensor:
    """ Computes collisions amongst ground truth.

        Inputs
        ------
            Y_hat[torch.tensor(B, 1, T, D)]: ego agent's ground truth
            Y_gt[torch.tensor(B, A, T, D)]: ground truth scene.
            num_agents[torch.tensor(B)]: number of agents in each scene.
            ego_agent[torch.tensor(B)]: ID of ego agent within the scene.

        Output
        ------
            collisions[torch.tensor(B)]: worst mode's (max. number of) collisions.

        """

    B, A, T, D = Y_gt.shape
    Y_hat = Y_hat[..., -T:, :, :].cpu().numpy()
    Y_gt = Y_gt.cpu().numpy()



    collisions = torch.zeros(size=(B,))

    for b in range(B):
        ego_gt = Y_hat[b, 0]                          # 1, T, D
        mask = np.zeros(shape=(A,), dtype=bool)        # A
        mask[:num_agents[b]] = True
        mask[ego_agent[b]] = False
        other_Y = Y_gt[b, mask]                        # A-1, T, D
        collisions[b] = max(
            [compute_collision(ego_gt, other_Y, coll_thresh)])
        return collisions.to('cuda:0')