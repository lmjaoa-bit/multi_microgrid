import numpy as np
from torch.nn import functional as F


class TeacherBudget:
    def __init__(self, fraction, max_solve_seconds):
        self.fraction, self.max_solve_seconds = float(fraction), float(max_solve_seconds)
        if not 0 < self.fraction <= .15 or not np.isfinite(self.max_solve_seconds) or self.max_solve_seconds <= 0:
            raise ValueError('Teacher budget requires fraction in (0,.15] and positive solve limit')
        self.seconds, self.calls, self.mean_seconds = 0., 0, 0.

    def remaining(self, nonteacher_seconds):
        return max(0., self.fraction*max(0.,nonteacher_seconds)-self.seconds)

    def admit(self, nonteacher_seconds):
        return self.remaining(nonteacher_seconds) >= max(.001, self.mean_seconds)

    def record(self, seconds):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid elapsed time')
        self.seconds += seconds
        self.calls += 1
        self.mean_seconds = self.seconds/self.calls


def plan_supervision_loss(output, batch, battery):
    mask = (output.mask & batch['mask']).to(output.intents.dtype)
    def mean(error):
        return ((error*mask).sum(-1)/mask.sum(-1).clamp_min(1.)).mean()
    scale = max(float(battery['max_charge_kw']),float(battery['max_discharge_kw']))
    intent = mean(F.smooth_l1_loss(output.intents,batch['intents'],reduction='none'))
    power = mean(F.smooth_l1_loss(output.request_kw/scale,batch['request_kw']/scale,reduction='none'))
    soc = mean(F.smooth_l1_loss(output.nominal_soc_after,batch['nominal_soc_after'],reduction='none'))
    return intent+power+soc, dict(plan_intent_loss=intent.detach(),plan_power_loss=power.detach(),plan_soc_loss=soc.detach())
