import numpy as np
import sympy as sp
from numpy.linalg import inv

x = sp.symbols('x')
y = sp.symbols('y')


def width(points):
    n = len(points) - 1
    distance = [np.sqrt((points[0][0] - points[n][0]) ** 2 + (points[0][1] - points[n][1]) ** 2)]

    for i in range(0, n):
        distance.append(np.sqrt((points[i][0] - points[i+1][0]) ** 2 + (points[i][1] - points[i+1][1]) ** 2))
    return np.amax(distance)


def simplex_method(n=0, e=1.0, guess=np.array([[2.0, 2.0, 0.0], [5.0, 5.0, 0], [-3.0, -3.0, 0.0]])
                   , f=100*(y-x**2)**2 + (1 - x)**2):
    if n == 0:
        for i in range(0, len(guess)):
            guess[i][2] = f.evalf(subs={x: guess[i][0], y: guess[i][1]})

    if e < 0.000001:
        print("From simplex method. Number of iteration: " + str(n))
        print("Maximizing values")
        return np.round(guess[0][0:2], 2)

    else:
        guess = guess[guess[:, 2].argsort()]

        reflection = guess.copy()
        for i in range(0, len(guess)):
            guess_copy = guess.copy()
            guess_copy = np.delete(guess_copy, i, 0)
            reflection[i] = (2 * guess[i] - np.mean(guess_copy, axis=0))

            reflection[i][2] = f.evalf(subs={x: reflection[i][0], y: reflection[i][1]})

        reflection_diff = []
        for i in range(0, len(guess)):
            reflection_diff.append(reflection[i][2] - guess[i][2])

        reflection_diff = np.array(reflection_diff)
        contraction = guess.copy()

        if np.all(reflection_diff <= 0):
            guess = reflection

        elif np.all(reflection_diff > 0):
            for i in range(0, len(guess)):
                guess_copy = guess.copy()
                guess_copy = np.delete(guess_copy, [i], axis=0)
                contraction[i] = (np.mean(guess_copy, axis=0))
                contraction[i][2] = f.evalf(subs={x: contraction[i][0], y: contraction[i][1]})
            guess = contraction

        else:
            for i in range(0, len(guess)):
                if reflection[i][2] <= guess[i][2]:
                    guess[i] = reflection[i]

        e = width(guess)
        n = n + 1
        return simplex_method(n=n, e=e, guess=guess, f=f)


class GuessObject:

    def __init__(self, guess=[2., 1.], f=100*(y-x**2)**2 + (1 - x)**2, hess=None):

        self.guess = guess
        self.f = f
        self.gradient = [sp.diff(f, x).evalf(subs={x: guess[0], y: guess[1]}),
                         sp.diff(f, y).evalf(subs={x: guess[0], y: guess[1]})]
        if hess is None:
            self.hess = [[round(sp.diff(f, x, x).evalf(subs={x: guess[0], y: guess[1]}), 2),
                          round(sp.diff(f, x, y).evalf(subs={x: guess[0], y: guess[1]}), 2)],

                         [round(sp.diff(f, y, x).evalf(subs={x: guess[0], y: guess[1]}), 2),
                          round(sp.diff(f, y, y).evalf(subs={x: guess[0], y: guess[1]}), 2)]
                         ]
        else:
            self.hess = hess

    def newtons_method(self, e=1.0, delta=1.0, n=0):

        if (e < 0.000001) and (delta < 0.000001):
            result = [round(self.guess[0], 2), round(self.guess[1], 2)]
            print("From  newtowns method. Number of iteration: " + str(n))
            print("Maximizing values")
            return result
        else:

            delta = np.abs(self.gradient[0]) + np.abs(self.gradient[1])
            guess_new = self.guess - inv(np.array(self.hess)).dot(np.array(self.gradient).transpose())
            guess_new_obj = GuessObject(guess=guess_new, f=self.f)
            e = np.maximum(np.abs(self.guess[0] - guess_new[0]), np.abs(self.guess[1] - guess_new[1]))

            return guess_new_obj.newtons_method(e=e, delta=delta, n=n+1)

    def min_line_search(self, sk_guess):

        min_array = np.zeros((len(sk_guess) + 1, len(sk_guess) + 1))

        sk_guess.sort()

        if (sk_guess[1] - sk_guess[0]) < (sk_guess[2] - sk_guess[1]):
            sk_guess.insert(3, (sk_guess[2] + sk_guess[1]) / 2.0)
        else:
            sk_guess.insert(3, (sk_guess[0] + sk_guess[1]) / 2.0)

        for i in range(0, len(sk_guess)):
            internal_guess = self.guess - inv(np.array(self.hess)).dot(np.array(self.gradient).transpose().dot(sk_guess[i]))

            min_array[i] = [sk_guess[i], internal_guess[0], internal_guess[1], self.f.evalf(subs={x: internal_guess[0],
                                                                                             y: internal_guess[1]})]
        if min_array[3][0] < min_array[1][0]:
            if min_array[3][3] > min_array[1][3]:
                min_array = np.delete(min_array, 0, 0)
                sk_guess = np.delete(sk_guess, 0)
            else:
                min_array = np.delete(min_array, 2, 0)
                sk_guess = np.delete(sk_guess, 2)

        elif min_array[3][0] > min_array[1][0]:
            if min_array[3][3] < min_array[1][3]:
                min_array = np.delete(min_array, 0, 0)
                sk_guess = np.delete(sk_guess, 0)
            else:
                min_array = np.delete(min_array, 2, 0)
                sk_guess = np.delete(sk_guess, 2)

        sk_guess.sort()

        if np.abs(sk_guess[0] - sk_guess[2]) < 0.1:
            return min_array[0][1:3]

        else:
            return self.min_line_search(sk_guess=sk_guess.tolist())

    def newtons_line_search_method(self, e=1.0, delta=1.0, n=0):

        if (e < 0.0001) and (delta < 0.0001):
            result = [round(self.guess[0], 2), round(self.guess[1], 2)]
            print("From  newtowns method with line search. Number of iteration: " + str(n))
            print("Maximizing values")
            return result

        else:
            delta = np.abs(self.gradient[0]) + np.abs(self.gradient[1])
            guess_new = self.min_line_search(sk_guess=[-5.0, 1.0, 5.0])
            guess_new_obj = GuessObject(guess=guess_new, f=self.f)
            e = np.maximum(np.abs(self.guess[0] - guess_new[0]), np.abs(self.guess[1] - guess_new[1]))

            return guess_new_obj.newtons_line_search_method(e=e, delta=delta, n=n+1)

    def bfgs_method(self, e=1.0, delta=1.0):

        print self.guess

        print self.hess
        if (e < 0.0001) and (delta < 0.0001):
            guess = [round(self.guess[0]), round(self.guess[1])]
            return guess

        else:
            delta = np.abs(self.gradient[0]) + np.abs(self.gradient[1])
            guess_new = self.guess - inv(np.array(self.hess)).dot(np.array(self.gradient).transpose())
            guess_new_obj = GuessObject(guess=guess_new, f=self.f, hess=[[0, 0], [0, 0]])
            e = np.maximum(np.abs(self.guess[0] - guess_new[0]), np.abs(self.guess[1] - guess_new[1]))

            z_k = np.array(guess_new_obj.guess - self.guess)
            y_k = np.array(guess_new_obj.gradient) - np.array(self.gradient)

            mid_term = np.array(self.hess).dot(z_k) * (z_k.transpose()).dot(np.array(self.hess)) / \
                       z_k.transpose().dot(np.array(self.hess)).dot(z_k)

            new_hess = self.hess - mid_term + (y_k.dot(y_k.transpose()) / y_k.transpose().dot(z_k))

            a = np.round(np.array(new_hess, dtype=np.float32))
            for i in range(0, 2):
                guess_new_obj.hess[i][0] = a[i][0]
                guess_new_obj.hess[i][1] = a[i][1]

            return guess_new_obj.bfgs_method(e=e, delta=delta)

if __name__ == '__main__':
    simplex_guess = np.array([[2.0, 2.0, 0.0], [5.0, 5.0, 0], [-3.0, -3.0, 0.0]])

    print(simplex_method(guess=simplex_guess))

    newton_guess = [3., 4.]

    new_guess_object = GuessObject(guess=newton_guess)
    print(" ")
    print(new_guess_object.newtons_method())

    print(" ")
    print(new_guess_object.newtons_line_search_method())



