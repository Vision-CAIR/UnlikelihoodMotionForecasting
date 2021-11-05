import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def get_traj_direction(traj):
    traj_pad = nn.functional.pad(traj, [0, 0, 1, 1])
    traj_pad[..., 0, :] = 2 * traj_pad[..., 1, :] - traj_pad[..., 2, :]
    traj_pad[..., -1, :] = 2 * traj_pad[..., -2, :] - traj_pad[..., -3, :]

    appo_vel = (traj_pad[..., 2:, :] - traj_pad[..., :-2, :]) / 2
    appo_vel_norm = torch.norm(appo_vel, dim=-1, keepdim=True)
    traj_dir = appo_vel / appo_vel_norm

    # mask out the points where speed is too low
    traj_dir = traj_dir * (appo_vel_norm >= 2).float()

    return traj_dir


def safety_checker(traj, safe_map):
    traj_shape = traj.shape
    n_batch = traj_shape[-3]
    _, _, h, w = safe_map.shape

    # lane_info_map = torch.norm(safe_map[:, :2], dim=-1)

    traj_direction = get_traj_direction(traj)

    traj_round = torch.round(traj).long()
    # the points outside the map
    out_idx = (traj_round[..., 0] >= h) + (traj_round[..., 1] >= w) + (traj_round < 0).sum(dim=-1).type(torch.bool)
    traj_round[out_idx] = 0

    idx = torch.cat([torch.zeros_like(traj_round)[..., :1], traj_round], dim=-1)
    idx[..., 0] = torch.arange(0, n_batch, device=traj_round.device)[:, None]
    idx = idx.view(-1, 3)

    # check lane direction
    lane_direction = safe_map[idx[:, 0], :2, idx[:, 1], idx[:, 2]].view(*traj_shape[:-1], 2).flip(dims=[-1])
    check_lane = torch.sum(traj_direction * lane_direction, dim=-1) < 0

    # check drivable region
    drivable_region = safe_map[idx[:, 0], 2, idx[:, 1], idx[:, 2]].view(*traj_shape[:-1])
    check_driable = drivable_region < 1

    safety_step = (check_lane + check_driable) * ~out_idx  # True means no safe
    safety = safety_step.sum(-1).bool()
    safety_step = safety_step.bool()

    return safety, safety_step


def visualization_unsafe(traj_total, safe_map_total, safety, max=20):
    if len(traj_total.shape) == 4:
        sample_list, batch_list = torch.where(safety)
        idx_list = zip(batch_list, sample_list)
    else:
        batch_list = torch.where(safety)[0]
        idx_list = [[i] for i in batch_list]

    for i, idx in enumerate(idx_list):
        visualization(traj_total, safe_map_total, *idx)
        if i >= max-1: break


def visualization(traj_total, safe_map_total, batch_idx, sample_idx=0):
    if len(traj_total.shape) == 4:
        traj = traj_total[sample_idx, batch_idx].detach().cpu()
    else:
        traj = traj_total[batch_idx].detach().cpu()
    safe_map = safe_map_total[batch_idx].detach().cpu()
    visualization_lane_direction(safe_map)
    plot_traj(traj)
    plt.show()


def visualization_lane_direction(safe_mask):
    direction_mask = safe_mask[:2].permute(1, 2, 0)
    drivable_mask = safe_mask[2]
    norm_x = np.round(np.linalg.norm(direction_mask, axis=-1))
    mask = np.stack([np.zeros_like(norm_x), drivable_mask, norm_x], axis=-1)
    plt.imshow(mask, origin='lower')
    a, b = np.where(norm_x > 0)
    select_idx = np.random.randint(0, len(a), 20) if len(a) > 0 else []
    for idx in select_idx:
        plt.arrow(b[idx], a[idx], direction_mask[a[idx], b[idx]][0] * 5, direction_mask[a[idx], b[idx]][1] * 5,
                  color='r', head_width=2)


def plot_traj(traj):
    appo_vel = get_traj_direction(traj)
    plt.plot(traj[:, 1], traj[:, 0], 'o-b', markersize=3)
    for i in range(len(traj)):
        if np.linalg.norm(appo_vel[i]) == 0: continue
        plt.arrow(traj[i, 1], traj[i, 0], appo_vel[i, 1] * 5, appo_vel[i, 0] * 5,
                  color='y', head_width=2)
