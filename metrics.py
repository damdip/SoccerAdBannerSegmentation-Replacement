import numpy as np

# Source: https://github.com/sacmehta/ESPNet/blob/master/train/IOUEval.py

class IOUEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 1

    def fast_hist(self, a, b, chunk_size=1000000):
        """
        Compute histogram in chunks to avoid memory issues
        """
        k = (a >= 0) & (a < self.nClasses)
        a = a[k]
        b = b[k]
        total_size = len(a)
        hist = np.zeros((self.nClasses, self.nClasses), dtype=np.float32)
        
        for i in range(0, total_size, chunk_size):
            end = min(i + chunk_size, total_size)
            chunk_hist = np.bincount(
                self.nClasses * a[i:end].astype(int) + b[i:end],
                minlength=self.nClasses ** 2
            ).reshape(self.nClasses, self.nClasses)
            hist += chunk_hist
            
        return hist

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        predict = predict.cpu().numpy().flatten()
        gth = gth.cpu().numpy().flatten()

        epsilon = 0.00000001
        hist = self.compute_hist(predict, gth)
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIou = np.nanmean(per_class_iu)

        self.overall_acc +=overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.mIOU += mIou
        self.batchCount += 1

    def getMetric(self):
        overall_acc = self.overall_acc/self.batchCount
        per_class_acc = self.per_class_acc / self.batchCount
        per_class_iu = self.per_class_iu / self.batchCount
        mIOU = self.mIOU / self.batchCount

        return overall_acc, per_class_acc, per_class_iu, mIOU