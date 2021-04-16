
# Logistic Class
# lm_shift= -0.5
# lm_scale = 2
class LM:
    def __init__(self, x_0, r, shift, scale):
        self.x = x_0
        self.r = r
        self.shift= shift
        self.scale = scale

    def next_val(self):
        self.x = self.r * self.x *(1.0-self.x)
        return self.x

    def shift_scale_next(self):
        return (self.next_val() + self.shift) * self.scale







# Gaussian class - for GA and mutating R
class Gauss:
    def __init__(self, shift, scale):
        self.shift= shift
        self.scale = scale

    def next_val(self):
        return np.random.normal()

    #Shift affects the mean, and scale affects the standard deviation!
    def shift_scale_next(self):
        return (self.next_val() + self.shift) * self.scale
