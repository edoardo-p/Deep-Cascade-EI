import numpy as np
from utils.metric import cal_mse, cal_psnr


def closure_ei(
    net,
    dataloader,
    physics,
    transform,
    optimizer,
    criterion_mc,
    criterion_ei,
    alpha,
    dtype,
    report_psnr=False,
):
    loss_mc_seq, loss_ei_seq, loss_seq, psnr_seq, mse_seq = [], [], [], [], []
    for x in dataloader:
        x = x[0] if isinstance(x, list) else x
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(dtype)  # ground-truth signal x

        y0 = physics.A(x.type(dtype))  # generate measurement input y
        x0 = physics.A_dagger(y0)  # range input (A^+y)

        x1 = net(x0)
        y1 = physics.A(x1)

        # equivariant imaging: x2, x3
        x2 = transform.apply(x1)
        x3 = net(physics.A_dagger(physics.A(x2)))

        loss_mc = criterion_mc(y1, y0)
        loss_ei = criterion_ei(x3, x2)

        loss = loss_mc + alpha["ei"] * loss_ei

        loss_mc_seq.append(loss_mc.item())
        loss_ei_seq.append(loss_ei.item())
        loss_seq.append(loss.item())

        if report_psnr:
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_closure = {
        "mc": np.mean(loss_mc_seq),
        "ei": np.mean(loss_ei_seq),
        "total": np.mean(loss_seq),
    }

    if report_psnr:
        loss_closure["psnr"] = np.mean(psnr_seq)
        loss_closure["mse"] = np.mean(mse_seq)

    return loss_closure
