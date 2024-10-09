import numpy as np
import random
import math

# SMO algorithm class
class SMO():
    def __init__(self, objf, lb, ub, dim, PopSize, acc_err, iters):
        self.PopSize = PopSize
        self.dim = dim
        self.acc_err = acc_err
        self.lb = np.array(lb)    # Convert to numpy array
        self.ub = np.array(ub)    # Convert to numpy array
        self.objf = objf
        self.pos = np.zeros((PopSize, dim))
        self.fun_val = np.zeros(PopSize)
        self.fitness = np.zeros(PopSize)
        self.prob = np.zeros(PopSize)
        self.LocalLimit = dim * PopSize
        self.GlobalLimit = PopSize
        self.fit = np.zeros(PopSize)
        self.MinCost = np.zeros(iters)
        self.Bestpos = np.zeros(dim)
        self.group = 0
        self.func_eval = 0
        self.part = 1
        self.max_part = 5
        self.cr = 0.1

    def CalculateFitness(self, fun1):
        return 1 / (fun1 + 1) if fun1 >= 0 else 1 + math.fabs(fun1)

    def initialize(self):
        global GlobalMin, GlobalLeaderPosition, GlobalLimitCount, LocalMin, LocalLimitCount, LocalLeaderPosition
        S_max = int(self.PopSize / 2)
        LocalMin = np.zeros(S_max)
        LocalLeaderPosition = np.zeros((S_max, self.dim))
        LocalLimitCount = np.zeros(S_max)
        for i in range(self.PopSize):
            for j in range(self.dim):
                self.pos[i, j] = random.uniform(self.lb[j], self.ub[j]) if isinstance(self.lb, np.ndarray) else random.uniform(self.lb, self.ub)
        for i in range(self.PopSize):
            self.pos[i, :] = np.clip(self.pos[i, :], self.lb, self.ub)
            self.fun_val[i] = self.objf(self.pos[i, :])
            self.func_eval += 1
            self.fitness[i] = self.CalculateFitness(self.fun_val[i])
        GlobalMin = self.fun_val[0]
        GlobalLeaderPosition = self.pos[0, :]
        GlobalLimitCount = 0

    def create_group(self):
        g = 0
        lo = 0
        self.gpoint = np.zeros((self.PopSize, 2))
        while lo < self.PopSize:
            hi = lo + int(self.PopSize / self.part)
            self.gpoint[g, 0] = lo
            self.gpoint[g, 1] = hi if (self.PopSize - hi) >= (int(self.PopSize / self.part)) else self.PopSize - 1
            g += 1
            lo = hi + 1
        self.group = g

    def CalculateProbabilities(self):
        maxfit = max(self.fitness)
        for i in range(self.PopSize):
            self.prob[i] = (0.9 * (self.fitness[i] / maxfit)) + 0.1

    def GlobalLearning(self):
        global GlobalMin, GlobalLeaderPosition
        G_trial = GlobalMin
        for i in range(self.PopSize):
            if self.fun_val[i] < GlobalMin:
                GlobalMin = self.fun_val[i]
                GlobalLeaderPosition = self.pos[i, :]

    def LocalLeaderPhase(self, k):
        lo = int(self.gpoint[k, 0])
        hi = int(self.gpoint[k, 1])
        for i in range(lo, hi + 1):
            PopRand = random.randint(lo, hi)
            while PopRand == i:
                PopRand = random.randint(lo, hi)
            new_position = self.pos[i, :] + (GlobalLeaderPosition - self.pos[i, :]) * random.random() + (self.pos[PopRand, :] - self.pos[i, :]) * (random.random() - 0.5) * 2
            new_position = np.clip(new_position, self.lb, self.ub)
            ObjValSol = self.objf(new_position)
            self.func_eval += 1
            FitnessSol = self.CalculateFitness(ObjValSol)
            if FitnessSol > self.fitness[i]:
                self.pos[i, :] = new_position
                self.fun_val[i] = ObjValSol
                self.fitness[i] = FitnessSol

    def GlobalLeaderPhase(self, k):
        lo = int(self.gpoint[k, 0])
        hi = int(self.gpoint[k, 1])
        for i in range(lo, hi + 1):
            if random.random() < self.prob[i]:
                PopRand = random.randint(lo, hi)
                while PopRand == i:
                    PopRand = random.randint(lo, hi)
                param2change = random.randint(0, self.dim - 1)
                new_position = self.pos[i, :].copy()
                new_position[param2change] += (GlobalLeaderPosition[param2change] - self.pos[i, param2change]) * random.random() + (self.pos[PopRand, param2change] - self.pos[i, param2change]) * (random.random() - 0.5) * 2
                new_position = np.clip(new_position, self.lb, self.ub)
                ObjValSol = self.objf(new_position)
                self.func_eval += 1
                FitnessSol = self.CalculateFitness(ObjValSol)
                if FitnessSol > self.fitness[i]:
                    self.pos[i, :] = new_position
                    self.fun_val[i] = ObjValSol
                    self.fitness[i] = FitnessSol

# Main optimization function
def exec_smo(objf1, lb1, ub1, dim1, PopSize1, iters, acc_err1, obj_val, succ_rate, mean_feval):
    smo = SMO(objf1, lb1, ub1, dim1, PopSize1, acc_err1, iters)
    smo.initialize()
    smo.create_group()

    for l in range(iters):
        for k in range(smo.group):
            smo.LocalLeaderPhase(k)
        smo.CalculateProbabilities()
        for k in range(smo.group):
            smo.GlobalLeaderPhase(k)
        smo.GlobalLearning()

        smo.cr = smo.cr + (0.4 / iters)
        smo.MinCost[l] = GlobalMin

        if math.fabs(GlobalMin - obj_val) <= smo.acc_err:
            succ_rate += 1
            mean_feval += smo.func_eval
            break

    return smo.MinCost, GlobalMin, GlobalLeaderPosition
