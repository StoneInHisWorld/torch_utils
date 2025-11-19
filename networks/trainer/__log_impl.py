import torch


@torch.no_grad()
def log_impl(
        pbar, preds, labels, ls_es, ls_names,
        criterion_a, criteria_names,
        metric_acc
):
    num_samples = len(preds)
    metrics = [criterion(preds, labels) for criterion in criterion_a]
    metric_acc.add(
        *[criterion(preds, labels) for criterion in criterion_a],
        *[ls * num_samples for ls in ls_es], num_samples
    )
    pbar.set_postfix(
        **{k: v.item() for k, v in zip(ls_names, ls_es)},
        **{k: v.item() for k, v in zip(criteria_names, metrics)}
    )
    pbar.update(1)
