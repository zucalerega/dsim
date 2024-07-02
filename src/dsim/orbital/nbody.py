import numpy as np
import matplotlib.pyplot as plt
import time
# UNIVERSAL CONSTANTS
G = 6.67430e-11
SCALING_FACTOR = 10e15
class Body:
    def __init__(self, x, v, m):
        self.x = np.array(x)
        self.v = np.array(v)
        self.m = m




class Craft:
    pass


class System:
    def __init__(self, *args):
        self.bodies = []
        for arg in args:
            self.bodies.append(arg)
        self.history = {}
        self.t = 0

    def add(self, body):
        self.bodies.append(body)

    def bodies2array(self):
        x_arr = np.ndarray([1, 3])
        v_arr = np.ndarray([1, 3])
        for i in self.bodies:
            np.vstack([x_arr, i.x])
            np.vstack([v_arr, i.v])
        return x_arr, v_arr

    def step(self, dt):
        bodies = self.bodies
        # big = max([b.m for b in bodies])
        # bodies = [b for b in bodies if b.m*10e3 > big]
        for body in self.bodies:
            body.x = body.x + dt * body.v
            a = np.zeros(3)
            for a_body in bodies:
                if body != a_body:
                    a = a - (((a_body.m * G) / np.abs(np.sum((body.x - a_body.x)**2))) * ((body.x - a_body.x) / np.sqrt(np.abs(np.sum((body.x - a_body.x)**2)))))
            body.v = body.v + dt * a
        self.t += dt
        self.history[self.t] = []
        for body in self.bodies:
            self.history[self.t].append([body, body.x, body.v])

    def plot(self, total=True):
            
        for body in range(len(list(self.history.values())[0])):
            positionsx = []
            positionsy = []
            for state in list(self.history.values()):
                positionsx.append(state[body][1][0])
                positionsy.append(state[body][1][1])
                
                plt.plot(positionsx, positionsy)
        plt.show()

    def __repr__(self):
        bodies = []
        for bod in self.bodies:
            body = []
            body.append([bod.x, bod.v, bod.m])
            bodies.append(body)
        return str(bodies)
    


# FUNCTIONS FOR SYSTEMS

def create_stable_system(type):
    if type == "single-star":
        planets = 5
        star_m = 10e30
        star = Body([0, 0, 0], [0, 0, 0], star_m)
        sys = System(star)

    elif type == "double-star":
        planets = 5
        star_m = 10e30
        star1 = Body([0, 0, 0], [0, 0, 0], star_m)
        star2 = Body([10e13, 0, 0], [0, 0, 0], star_m)

        sys = System(star1, star2)


    for i in range(planets):
        m = 10e24
        x = np.array([0, np.random.randint(1, 1000, dtype="int64"), 0]) * 10e9
        v = [np.sqrt((G * star_m) / np.sqrt(np.sum(x*x))), 0, 0]
        sys.add(Body(x, v, m))
    
    return sys


# bod1 = Body(np.array([1e12, 1e12, 0]), np.array([1e2, 0, 0]), 2e30)
# bod2 = Body(np.array([1e6, 1e6, 0]), np.array([0, 1e4, 0]), 2e25)
# bod3 = Body(np.array([0, 1e12, 0]), np.array([1e4, 1e4, 0]), 2e25)
# bod4 = Body(np.array([0, 1e12, 0]), np.array([1e4, 0, 0]), 2e25)
sys = create_stable_system("single-star")

# sys = System(bod1, bod2, bod3, bod4)
t = 0
t1 = time.time()
for i in range(2000):    
    t += 8000000
    sys.step(8000000)
print(f'Sim took: {round(time.time() - t1, 3)}s')
sys.plot()
