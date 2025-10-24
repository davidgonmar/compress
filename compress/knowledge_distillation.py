import torch
import torch.nn.functional as F


def knowledge_distillation_loss(
    logits_teacher: torch.Tensor, logits_student: torch.Tensor, T: float = 1.0
) -> torch.Tensor:
    student_log_probs = F.log_softmax(logits_student / T, dim=-1)
    teacher_probs = F.softmax(logits_teacher / T, dim=-1).detach()
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T**2)
    return loss
