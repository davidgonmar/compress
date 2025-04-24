import torch


def knowledge_distillation_loss(
    logits_teacher: torch.Tensor, logits_student: torch.Tensor, T: float = 1.0
) -> torch.Tensor:
    logits_teacher = logits_teacher / T
    logits_student = logits_student / T

    p_teacher = torch.nn.functional.softmax(logits_teacher, dim=-1)
    p_student = torch.nn.functional.softmax(logits_student, dim=-1)

    loss = torch.nn.functional.kl_div(
        p_student.log(), p_teacher, reduction="batchmean"
    ) * (T**2)
    return loss
