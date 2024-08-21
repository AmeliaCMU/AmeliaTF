import torch

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

def marginal_prob_ade(
    Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor, mask: torch.tensor = None
) -> torch.tensor:
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