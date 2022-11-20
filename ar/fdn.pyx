import numpy as np
import scipy
from scipy import signal

cdef class Biquad:
    cdef:
        double b0, b1, b2, a1, a2
        double s0 
        double s1

    def __cinit__(self, double b0, double b1, double b2, double a1, double a2):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.s0 = 0.
        self.s1 = 0.

    cpdef double filter(self, double x) except *:
        cdef double y = self.b0 * x + self.s0
        self.s0 = 1
        self.s0 = self.b1 * x - self.a1 * y + self.s1
        self.s1 = self.b2 * x - self.a2 * y
        return y

cdef class IIR:
    cdef:
        list sos

    def __cinit__(self, double[:,::1] Bs, double[:,::1] As):
        self.sos = [Biquad(b0 = Bs[i, 0] / As[i, 0],
                           b1 = Bs[i, 1] / As[i, 0],
                           b2 = Bs[i, 2] / As[i, 0],
                           a1 = As[i, 1] / As[i, 0],
                           a2 = As[i, 2] / As[i, 0]) for i in range(Bs.shape[0])]

    cpdef double filter(self, double x) except *:
        cdef double y = x
        cdef Py_ssize_t i
        for i in range(len(self.sos)):
            y = self.sos[i].filter(y)
        return y

cdef class SAP:
    cdef:
        double alpha
        int m
        cdef Py_ssize_t buffer_idx
        double[::1] delay_buffers

    def __cinit__(self, double alpha, int m):
        self.alpha = alpha
        self.m = int(m)
        self.delay_buffers = np.zeros(self.m, dtype = np.double)
        self.buffer_idx = 0

    cpdef filter(self, double x):
        cdef double y = self.delay_buffers[self.buffer_idx] + self.alpha * (x - self.alpha * self.delay_buffers[self.buffer_idx])
        self.delay_buffers[self.buffer_idx] = x - self.alpha * self.delay_buffers[self.buffer_idx]
        self.buffer_idx = (self.buffer_idx + 1) % self.m
        return y

cdef class APF:
    cdef:
        list saps

    def __cinit__(self, double[::1] alphas, int[::1] ms):
        self.saps = [SAP(alpha = alphas[i], m = ms[i]) for i in range(alphas.shape[0])]

    def filter(self, x):
        cdef double y = x
        cdef Py_ssize_t i
        for i in range(len(self.saps)):
            y = self.saps[i].filter(y)
        return y

cdef class CythonFDN:
    cdef:
        IIR pre_filter, post_filter
        list fb_filter, fb_apf
        double[:,:] Q, R
        double[::1] pre_gain, post_gain, fb_gain
        int[::1] ds
        Py_ssize_t num_channels
        bint time_varying

    def __cinit__(self, 
                  int[::1] ds,
                  double[::1] pre_gain, double[:,::1] pre_filter_Bs, double[:,::1] pre_filter_As,
                  double[::1] fb_adm, 
                  double[::1] fb_gain, double[:,:,::1] fb_filter_Bs, double[:,:,::1] fb_filter_As,
                  double[:,::1] fb_SAP_alphas, int[:,::1] fb_SAP_ms,
                  double[::1] post_gain, double[:,::1] post_filter_Bs, double[:,::1] post_filter_As,
                  bint time_varying):

        self.num_channels = len(ds)
        self.ds = ds
        self.pre_filter = IIR(pre_filter_Bs, As = pre_filter_As)
        self.post_filter = IIR(post_filter_Bs, As = post_filter_As)
        self.pre_gain = pre_gain
        self.post_gain = post_gain
        self.fb_gain = fb_gain
        self.fb_filter = []

        for i in range(len(ds)):
            self.fb_filter.append(IIR(fb_filter_Bs[i], fb_filter_As[i]))

        self.fb_apf = [APF(alphas = fb_SAP_alphas[i], ms = fb_SAP_ms[i]) for i in range(self.num_channels)]
        self.Q = 2 * np.matmul(fb_adm[:, None], fb_adm[None, :]) / np.sum(np.power(fb_adm, 2)) - np.eye(self.num_channels)
        self.time_varying = time_varying

        logMat = np.random.randn(self.num_channels, self.num_channels)
        skewsym = (logMat - logMat.T) / 2
        E, eigvec = np.linalg.eig(skewsym)
        nE = E / np.abs(E) * np.pi / 12000
        skewsym = np.real(np.matmul(np.matmul(eigvec, np.diag(nE)), np.conjugate(eigvec.T)))
        skewsym = (skewsym - skewsym.T) / 2

        self.R = scipy.linalg.expm(skewsym).astype(np.double)

        if time_varying:
            rdx = np.random.randint(12000, size = 1)[0]
            for rdi in range(rdx):
                self.Q = np.matmul(self.Q, self.R)

    cpdef double[:] filter(self, double[::1] x):
        cdef Py_ssize_t x_len = x.shape[0]
        cdef Py_ssize_t i, ch_i, ch_j
        cdef double[::1] y = np.zeros((x_len,), dtype = np.double)
        cdef double[::1] fb_outs = np.zeros(self.num_channels)
        cdef int max_delay = max(self.ds)
        cdef double[:,::1] delay_buffers = np.zeros((self.num_channels, max_delay), dtype = np.double)
        cdef Py_ssize_t[::1] buffer_idx = np.zeros((self.num_channels,), dtype = np.intp)
        cdef double[::1] feedback_outputs = np.zeros((self.num_channels,), dtype = np.double)
        cdef double post_sum = 0
        cdef double pre_out = 0

        for i in range(x_len):
            post_sum = 0.
            for ch_i in range(self.num_channels):
                post_sum += delay_buffers[ch_i][buffer_idx[ch_i]] * self.post_gain[ch_i]
            y[i] = self.post_filter.filter(post_sum)

            for ch_i in range(self.num_channels):
                feedback_outputs[ch_i] = self.fb_apf[ch_i].filter(self.fb_filter[ch_i].filter(delay_buffers[ch_i][buffer_idx[ch_i]] * self.fb_gain[ch_i]))
                #feedback_outputs[ch_i] = self.fb_filter[ch_i].filter(delay_buffers[ch_i][buffer_idx[ch_i]]*.99999)
            pre_out = self.pre_filter.filter(x[i]) 
            for ch_i in range(self.num_channels):
                delay_buffers[ch_i][buffer_idx[ch_i]] = pre_out * self.pre_gain[ch_i]

            for ch_i in range(self.num_channels):
                for ch_j in range(self.num_channels):
                    delay_buffers[ch_i][buffer_idx[ch_i]] += self.Q[ch_i][ch_j] * feedback_outputs[ch_j]

            for ch_i in range(self.num_channels):
                buffer_idx[ch_i] = (buffer_idx[ch_i] + 1) % self.ds[ch_i]

            if self.time_varying:
                self.Q = np.matmul(self.Q, self.R)

        #cdef double[::1] bypass = signal.convolve(self.bypass_FIR, x)[:x_len]
        #for i in range(x_len):
        #    y[i] = y[i] + bypass[i]
        return y
