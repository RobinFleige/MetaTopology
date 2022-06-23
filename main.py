import numpy as np
import sys
from jax import grad, jacfwd
from sympy import *
from scipy.optimize import fsolve
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

def f_2p2d_simple_x(vars):
    s, t, x, y = vars
    return x**2-s-t


def f_2p2d_simple_y(vars):
    s, t, x, y = vars
    return y-s

def f_2p2d_simple_z(vars):
    s, t, x, y = vars
    return 2*x



f_2p2d_simple = [f_2p2d_simple_x, f_2p2d_simple_y]


def f_2p2d_double_x(vars):
    s, t, x, y = vars
    return (x**2+t)*(y-1)-(x**2+s)*(y+1)


def f_2p2d_double_y(vars):
    s, t, x, y = vars
    return (y-1)*(y+1)


f_2p2d_double = [f_2p2d_double_x, f_2p2d_double_y]


def f_3p3d_circle_x(vars):
    s, t, u, x, y, z = vars
    return x**2+s**2+t**2+u*2-1


def f_3p3d_circle_y(vars):
    s, t, u, x, y, z = vars
    return y


def f_3p3d_circle_z(vars):
    s, t, u, x, y, z = vars
    return z


f_3p3d_circle = [f_3p3d_circle_x, f_3p3d_circle_y, f_3p3d_circle_z]


class VectorField:
    DETAIL = 5
    DELTA = 1/DETAIL
    VECTOR_SIZE_THRESHOLD = 0.000001
    MIN = -4
    MAX = 4

    def __init__(self, base_function, parameter_dimensions, space_dimensions):
        self.base_function = base_function
        self.parameter_dimensions = parameter_dimensions
        self.space_dimensions = space_dimensions
        self.dimensions = parameter_dimensions + space_dimensions

    def jacobi(self):
        # outer = function; inner = parameter
        jac_matrix = []
        for f in self.base_function:
            derivatives = grad(f)
            jac_matrix.append(derivatives)
        return jac_matrix

    def jacobi_values(self, vars):
        # outer = function; inner = parameter
        jac = self.jacobi()
        matrix = []
        for f in range(self.space_dimensions):
            values = list(jac[f](vars))
            value_list = []
            for i in range(len(values)):
                value_list.append(float(values[i]))
            matrix.append(value_list)
        return matrix

    # Only the Space Parts
    def fff(self, vars, fff_id):
        jac = self.jacobi()

        ids = []
        for d in range(self.space_dimensions*2-2):
            if d == self.space_dimensions - 1:
                ids.append(fff_id)
            ids.append((d + 1) % (self.space_dimensions)+self.parameter_dimensions)

        fff = []
        for fff_component in range(self.space_dimensions):
            matrix = []
            for x in range(self.space_dimensions):
                line = []
                values = jac[x](vars)
                for y in range(self.space_dimensions):
                    line.append(float(values[ids[fff_component + y]]))
                matrix.append(line)
            fff.append(pow(-1,(fff_component+1)*self.space_dimensions)*self.determinant(matrix))
        return fff

    # Only the Parameter Part
    def fff_cond(self, vars):
        jac = self.jacobi()

        matrix = []
        for x in range(self.space_dimensions):
            line = []
            values = jac[x](vars)
            for y in range(self.space_dimensions):
                line.append(values[self.parameter_dimensions+y])
            matrix.append(line)
        fff_cond = self.determinant(matrix)
        return fff_cond

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

    def newton_step(self, point):
        both = []
        for i in range(self.parameter_dimensions):
            both.append(point[i])
        mins = []
        maxs = []
        for i in range(self.space_dimensions):
            mins.append(point[i+self.parameter_dimensions]-self.DELTA)
            maxs.append(point[i+self.parameter_dimensions]+self.DELTA)
        return both, mins, maxs

    def critical_points(self, both, mins, maxs, recursion_depth):
        center = both + [(mins[i]+maxs[i])/2 for i in range(self.space_dimensions)]
        if self.vector_length([self.base_function[i](center) for i in range(self.space_dimensions)]) < self.VECTOR_SIZE_THRESHOLD or recursion_depth == sys.getrecursionlimit()-20:
            return [center]
        else:

            corners = []
            for i in range(pow(2, self.space_dimensions)):
                corner = []
                corner = corner + both
                for d in range(self.space_dimensions):
                    if i%pow(2, d + 1) < pow(2, d):
                        corner.append(mins[d])
                    else:
                        corner.append(maxs[d])
                corners.append(corner)

            positive_counts = []
            for i in range(self.space_dimensions):
                positive_counts.append(0)

            for corner in corners:
                value = [self.base_function[i](corner) for i in range(self.space_dimensions)]
                for i in range(self.space_dimensions):
                    if value[i] > 0:
                        positive_counts[i] = positive_counts[i] + 1

            contains_crit = True
            for i in range(self.space_dimensions):
                if positive_counts[i] == 0 or positive_counts[i] == len(corners):
                    contains_crit = False

            if contains_crit:
                new_mins = []
                new_maxs = []
                for i in range(pow(2, self.space_dimensions)):
                    new_min = []
                    new_max = []
                    for d in range(self.space_dimensions):
                        if i % pow(2, d + 1) < pow(2, d):
                            new_min.append(mins[d])
                            new_max.append((mins[d]+maxs[d])/2)
                        else:
                            new_min.append((mins[d]+maxs[d])/2)
                            new_max.append(maxs[d])
                    new_mins.append(new_min)
                    new_maxs.append(new_max)

                crits = []
                for i in range(len(new_mins)):
                    crits = crits + self.critical_points(both, new_mins[i], new_maxs[i], recursion_depth + 1)
                return crits
            else:
                return []

    def crit_line(self, both, mins, maxs):
        start = self.critical_points(both, mins, maxs, 0)
        start_id = []
        for i in range(self.dimensions):
            start_id.append(0)
        nexts = []
        ids = []
        ids.append(start_id)
        nexts.append([start[0], start_id])
        crits = []
        while len(nexts) > 0:
            crit = nexts[0]
            crits.append(crit[0])
            nexts.remove(crit)

            in_range = True
            for i in range(self.dimensions):
                if crit[0][i] < self.MIN or crit[0][i] > self.MAX:
                    in_range = False

            if in_range:
                fff_cond = self.fff_cond(crit[0])
                if abs(fff_cond) < self.VECTOR_SIZE_THRESHOLD:
                    fff_cond = self.VECTOR_SIZE_THRESHOLD
                #Only in one direction (last parameter)
                i = self.parameter_dimensions-1
                fff = self.fff(crit[0], i)

                next_point = []
                prev_point = []
                next_point = next_point + crit[0]
                prev_point = prev_point + crit[0]

                next_point[i] = float(next_point[i] + (fff_cond/self.DETAIL))
                prev_point[i] = float(prev_point[i] - (fff_cond/self.DETAIL))

                for j in range(self.space_dimensions):
                    next_point[j+self.parameter_dimensions] = float(next_point[j+self.parameter_dimensions] + (fff[j]/self.DETAIL))
                    prev_point[j+self.parameter_dimensions] = float(prev_point[j+self.parameter_dimensions] - (fff[j]/self.DETAIL))

                next_both, next_mins, next_maxs = self.newton_step(next_point)
                prev_both, prev_mins, prev_maxs = self.newton_step(prev_point)
                next_crit = self.critical_points(next_both, next_mins, next_maxs, 0)
                prev_crit = self.critical_points(prev_both, prev_mins, prev_maxs, 0)

                next_id = []
                next_id = next_id + crit[1]
                next_id[i] = next_id[i]+1
                prev_id = []
                prev_id = prev_id + crit[1]
                prev_id[i] = prev_id[i]-1

                if next_crit != [] and next_id not in ids:
                    nexts.append([next_crit[0], next_id])
                    ids.append(next_id)
                if prev_crit != [] and prev_id not in ids:
                    nexts.append([prev_crit[0], prev_id])
                    ids.append(prev_id)

        return crits



vf = VectorField(f_2p2d_simple, 2, 2)
m_vf = VectorField([vf.fff_cond, f_2p2d_simple_x, f_2p2d_simple_y], 1, 3)


bifurcations = m_vf.crit_line([1.0], [-1.5, -0.5, 0.5], [-0.5, 0.5, 1.5])
crits = []
print(len(bifurcations))
for i in range(len(bifurcations)):
    print(i)
    crits = crits + vf.crit_line([bifurcations[i][0], bifurcations[i][1]], [bifurcations[i][2]-0.5, bifurcations[i][3]-0.5], [bifurcations[i][2]+0.5, bifurcations[i][3]+0.5])


#mm_vf = VectorField([m_vf.fff_cond, vf.fff_cond, f_3p3d_circle_x, f_3p3d_circle_y, f_3p3d_circle_z], 1, 5)
#print(mm_vf.fff_cond((0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))


points = vtkPoints()
vertices = vtkCellArray()
polydata = vtkPolyData()
colors = vtkUnsignedCharArray()
colors.SetNumberOfComponents(4)
colors.SetNumberOfTuples(len(crits)+len(bifurcations))

for i in range(len(crits)):
    rgba = [255, 0, 0, 255]
    colors.InsertTuple(i, rgba)
    p = [crits[i][0], crits[i][1], crits[i][2]]
    pid = points.InsertNextPoint(p)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(pid)

for i in range(len(bifurcations)):
    rgba = [0, 255, 0, 255]
    colors.InsertTuple(i+len(crits), rgba)
    p = [bifurcations[i][0], bifurcations[i][1], bifurcations[i][2]]
    pid = points.InsertNextPoint(p)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(pid)


polydata.SetPoints(points)
polydata.SetVerts(vertices)
polydata.GetCellData().SetScalars(colors);
mapper = vtkPolyDataMapper()
mapper.SetInputData(polydata)
actor = vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(10)
renderer = vtkRenderer()
renderer.AddActor(actor)
renderWindow = vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderer.SetBackground([0, 0, 255])
interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)
style = vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(style)
renderWindow.Render()
interactor.Start()



#FFF korrekt(er):           http://vc.cs.ovgu.de/index.php?article_id=3&clang=0&bibtex_key=Theisel_2003_VisSym
#FFF Initial:               http://vc.cs.ovgu.de/files/publications/Archive/Theisel_2003_VisSym.pdf
#FFF Allgemein:             Sign nur bei ungerader Anzahl an Dimensionen (entweder über Determinante oder Gram-Schmidts)
#Calculate Orthogonal:      https://www.geometrictools.com/Documentation/OrthonormalSets.pdf