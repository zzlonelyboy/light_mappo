from pathlib import Path
import torch

def _load_agent_ckpt(trainer, agent_id, model_dir: str, device: torch.device):
    actor_p = Path(model_dir) / f"actor_agent{agent_id}.pt"
    critic_p = Path(model_dir) / f"critic_agent{agent_id}.pt"
    if not actor_p.exists() or not critic_p.exists():
        raise FileNotFoundError(f"缺少权重文件：{actor_p} 或 {critic_p}")
    try:
        actor_sd = torch.load(str(actor_p), map_location=device, weights_only=True)  # 新版更安全
        critic_sd = torch.load(str(critic_p), map_location=device, weights_only=True)
    except TypeError:
        actor_sd = torch.load(str(actor_p), map_location=device)  # 兼容旧版
        critic_sd = torch.load(str(critic_p), map_location=device)
    trainer.policy.actor.load_state_dict(actor_sd)
    trainer.policy.critic.load_state_dict(critic_sd)
    trainer.policy.actor.to(device).eval()
    trainer.policy.critic.to(device).eval()