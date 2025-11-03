import torch
from tqdm.auto import tqdm
import time
import numpy as np
from src.losses import calc_prob_uncertinty

tic, toc = (time.time, time.time)


def train(
    probe, device, train_loader, optimizer, epoch, loss_func,
    class_names=None, report=False, verbose_interval=5, layer_num=40,
    head=None, verbose=True, return_raw_outputs=False,
    one_hot=False, uncertainty=False, **kwargs,
):
    """Train one epoch of the probe."""
    assert (verbose_interval is None) or verbose_interval > 0, \
        "invalid verbose_interval, verbose_interval(int) > 0"
    starttime = tic()
    probe.train()

    loss_sum, correct, tot = 0, 0, 0
    preds, truths = [], []

    for batch_idx, batch in enumerate(train_loader):
        target = batch["age"].long().cuda()
        if one_hot:
            target = torch.nn.functional.one_hot(target, **kwargs).float()

        optimizer.zero_grad()
        if layer_num or layer_num == 0:
            act = batch["hidden_states"][:, layer_num,].to("cuda")
        else:
            act = batch["hidden_states"].to("cuda")

        output = probe(act)

        # --- FIXED: handle BCELoss shape + dtype ---
        if isinstance(loss_func, torch.nn.BCELoss):
            loss = loss_func(output[0].squeeze(), target.float())
        else:
            loss = loss_func(output[0], target, **kwargs)

        loss.backward()
        optimizer.step()

        loss_sum += loss.sum().item()
        if uncertainty:
            pred, _ = calc_prob_uncertinty(output[0].detach().cpu().numpy())
        pred = torch.argmax(output[0], axis=1)

        if len(target.shape) > 1:
            target = torch.argmax(target, axis=1)

        correct += np.sum(pred.cpu().numpy() == target.cpu().numpy())
        if return_raw_outputs:
            preds.append(pred.cpu().numpy())
            truths.append(target.cpu().numpy())
        tot += pred.shape[0]

    train_acc = correct / tot
    loss_avg = loss_sum / len(train_loader)

    endtime = toc()
    if verbose:
        print(
            f"\nTrain set: Average loss: {loss_avg:.4f} "
            f"({endtime-starttime:.3f} sec) Accuracy: {train_acc:.3f}\n"
        )

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    if return_raw_outputs:
        return loss_avg, train_acc, preds, truths
    else:
        return loss_avg, train_acc


def test(
    probe, device, test_loader, loss_func,
    return_raw_outputs=False, verbose=True,
    layer_num=40, scheduler=None,
    one_hot=False, uncertainty=False, **kwargs,
):
    """Evaluate the probe on the test set."""
    probe.eval()
    test_loss, tot, correct = 0, 0, 0
    preds, truths = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            target = batch["age"].long().cuda()
            if one_hot:
                target = torch.nn.functional.one_hot(target, **kwargs).float()

            if layer_num or layer_num == 0:
                act = batch["hidden_states"][:, layer_num,].to("cuda")
            else:
                act = batch["hidden_states"].to("cuda")

            output = probe(act)

            # --- FIXED: handle BCELoss shape + dtype ---
            if isinstance(loss_func, torch.nn.BCELoss):
                loss = loss_func(output[0].squeeze(), target.float())
            else:
                loss = loss_func(output[0], target, **kwargs)

            test_loss += loss.sum().item()

            if uncertainty:
                pred, _ = calc_prob_uncertinty(output[0].detach().cpu().numpy())
            pred = torch.argmax(output[0], axis=1)

            if len(target.shape) > 1:
                target = torch.argmax(target, axis=1)

            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            correct += np.sum(pred == target)
            tot += pred.shape[0]

            if return_raw_outputs:
                preds.append(pred)
                truths.append(target)

    test_loss /= len(test_loader)
    if scheduler:
        scheduler.step(test_loss)

    test_acc = correct / tot
    if verbose:
        print(f"Test set: Average loss: {test_loss:.4f},  Accuracy: {test_acc:.3f}\n")

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    if return_raw_outputs:
        return test_loss, test_acc, preds, truths
    else:
        return test_loss, test_acc
