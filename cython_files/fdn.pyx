from __future__ import print_function

def fib(n):
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b
    print()




cdef class Biquad:
    cdef float b0, b1, b2, a1, a2, state_0, state_1
    def __init__(self, b0, b1, b2, a0, a1, a2):
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
        self.state_0 = 0.
        self.state_1 = 0.

    def filter(self, x):
        y = self.b0 * x + self.state_0
        self.state_0 = self.b1 * x - self.a1 * y + self.state_1
        self.state_1 = self.b2 * x - self.a2 * y
        return y

cdef class SAP:
    cdef float alpha, circ_buffer[500]
    cdef int m, buffer_idx

    def __init__(alpha, m):
        self.alpha = alpha
        self.m = m
        self.circ_buffer


cdef float biquad(float x, float* state_0, float* state_1, float b0, float b1, float b2, float a1, float a2):
    y = b0 * x + state_0
    state_0 = b1 * x - a1 * y + state_1
    state_1 = b2 * x - a2 * y
    return y

cdef float sap(float x, float alpha, float buffer_read):
    y = - alpha * (x + alpha * buffer_read) + buffer_read
    buffer_write = x + alpha * buffer_read
    return y, buffer_write

cdef 


