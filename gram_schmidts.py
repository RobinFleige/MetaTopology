from sympy import *
import random as r

class VectorField:
    def __init__(self, space_dimensions, jacobi):
        self.parameter_dimensions = 1
        self.space_dimensions = space_dimensions
        self.dimensions = self.parameter_dimensions + self.space_dimensions
        self.jacobi = jacobi

    def fff_by_det(self):

        ids = []
        for d in range(self.space_dimensions*2-1):
            if d == self.space_dimensions:
                ids.append(0)
            ids.append(d % (self.space_dimensions)+1)

        matrices = []
        for fff_component in range(self.space_dimensions+self.parameter_dimensions):
            matrix = []
            for x in range(self.space_dimensions):
                line = []
                values = jacobi[x]
                for y in range(self.space_dimensions):
                    line.append(values[ids[fff_component + y]])
                matrix.append(line)
            matrices.append(matrix)

        fff = []
        for fff_component in range(len(matrices)):
            sign = pow(-1, fff_component*self.space_dimensions)
            value = self.determinant(matrices[fff_component])
            fff.append(sign*value)
        return fff

    def determinant(self, matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        else:
            value = 0
            for i in range(len(matrix)):
                next_matrix = []
                for x in range(1, len(matrix)):
                    next_line = []
                    for y in range(len(matrix)):
                        if y != i:
                            next_line.append(matrix[x][y])
                    next_matrix.append(next_line)
                value += matrix[0][i] * pow(-1, i) * self.determinant(next_matrix)
        return value

    def vector_length(self, vector):
        value = 0
        for i in range(self.space_dimensions):
            value += vector[i]**2
        return sqrt(value)

    #list of vectors to cross
    def cross_product(self, vectors):
        ids = []
        for d in range(self.space_dimensions + 1):
            ids.append((d+1) % (self.space_dimensions))

        output = []
        for i in range(len(vectors[0])):
            output.append(vectors[0][ids[i]]*vectors[1][ids[i+1]]-vectors[0][ids[i+1]]*vectors[1][ids[i]])

        return output


    #list of vectors to cross
    def scalar_product(self, vectors):
        value = 0
        for i in range(len(vectors[0])):
            value += vectors[0][i]*vectors[1][i]
        return value

    def grad(self, id):
        return self.jacobi[id]

    def orthogonalize(self):
        self.jacobi = self.gram_schmidts(self.jacobi)

    def normalize(self):
        for j in range(len(self.jacobi)):
            vector = self.jacobi[j]
            value = 0
            for i in range(len(vector)):
                value = value + vector[i]

            for i in range(len(vector)):
                vector[i] = vector[i] / value

    def gram_schmidts(self,matrix):
        new_matrix = []
        new_matrix.append(matrix[0])
        for i in range(1, len(matrix)):
            vector = matrix[i].copy()
            for j in range(0, i):
                factor = self.scalar_product([new_matrix[j], matrix[i]]) / self.scalar_product([new_matrix[j], new_matrix[j]])
                new_vector = []
                for k in range(len(vector)):
                    new_vector.append(vector[k] - factor * new_matrix[j][k])
                vector = new_vector
            new_matrix.append(vector)
        return new_matrix

    def fff_by_gram(self):
        matrix = self.jacobi.copy()
        vector = []
        for i in range(len(matrix[0])):
            vector.append(1)
        matrix.append(vector)
        new_matrix = self.gram_schmidts(matrix)

        return new_matrix[len(new_matrix)-1]

def random_array(space_dims):
    matrix = []
    for i in range(space_dims):
        vector = []
        for j in range(space_dims+1):
            vector.append(floor(r.random()*20))
        matrix.append(vector)
    return matrix


space_dims = 3
jacobi = random_array(space_dims)#[[0,0,2,0],[-1,-1,0,0],[-1,0,0,1]]#
print(jacobi)
vf = VectorField(space_dims, jacobi)
#vf.orthogonalize()
#print(vf.jacobi)
#vf.normalize()
#print(vf.jacobi)

print("Det")
fff = vf.fff_by_det()
print("FFF: " + str(fff))
for i in range(space_dims):
    grad = vf.grad(i)
    dot = vf.scalar_product([fff, grad])
    print("Dot_" + str(i) + ": " + str(dot))


print("Gram-Schmidts")
fff = vf.fff_by_gram()
print(fff)
for i in range(space_dims):
    grad = vf.grad(i)
    dot = vf.scalar_product([fff, grad])
    print("Dot_" + str(i) + ": " + str(dot))

