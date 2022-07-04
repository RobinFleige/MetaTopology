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
    vtkPolyData,
    vtkPolyLine
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
    return x**2+s**2+t**2+u**2-1


def f_3p3d_circle_y(vars):
    s, t, u, x, y, z = vars
    return y


def f_3p3d_circle_z(vars):
    s, t, u, x, y, z = vars
    return z


f_3p3d_circle = [f_3p3d_circle_x, f_3p3d_circle_y, f_3p3d_circle_z]


def f_2p2d_circle_x(vars):
    s, t, x, y = vars
    return x**2+s**2+t**2-1


def f_2p2d_circle_y(vars):
    s, t, x, y = vars
    return y


def f_2p2d_circle_z(vars):
    s, t, x, y = vars
    return 2*x


f_2p2d_circle = [f_2p2d_circle_x, f_2p2d_circle_y]

class CritLine:
    def __init__(self):
        self.lower_order_critlines = []
        self.bifurcation_point = []
        self.crit_list = []

class VectorField:
    DETAIL = 10
    DELTA = 2/DETAIL
    VECTOR_SIZE_THRESHOLD = 0.0001
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
            fff.append(pow(-1, (fff_component+1)*self.space_dimensions)*self.determinant(matrix))
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

    def critical_points(self, both, mins, maxs, recursion_depth, min_recursion):
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
                for i in range(self.space_dimensions):
                    value = self.base_function[i](corner)
                    if value > 0:
                        positive_counts[i] = positive_counts[i] + 1

            contains_crit = True
            for i in range(self.space_dimensions):
                if positive_counts[i] == 0 or positive_counts[i] == len(corners):
                    contains_crit = False

            if contains_crit or min_recursion > 0:
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
                    crits = crits + self.critical_points(both, new_mins[i], new_maxs[i], recursion_depth + 1, min_recursion - 1)
                return crits
            else:
                return []

    def fff_step(self, point, direction):
        fff_cond = float(self.fff_cond(point))
        fff = self.fff(point, self.parameter_dimensions - 1)

        sum = abs(fff_cond)
        for i in range(len(fff)):
            sum = sum + abs(fff[i])

        if sum != 0:
            fff_cond = fff_cond / sum
            for i in range(len(fff)):
                fff[i] = fff[i] / sum

        next_point = point.copy()
        next_point[self.parameter_dimensions - 1] = float(point[self.parameter_dimensions - 1] + (fff_cond / self.DETAIL) * pow(-1, direction))

        for j in range(self.space_dimensions):
            next_point[j + self.parameter_dimensions] = float(
                next_point[j + self.parameter_dimensions] + (fff[j] / self.DETAIL) * pow(-1, direction))

        next_both, next_mins, next_maxs = self.newton_step(next_point)
        next_crit = self.critical_points(next_both, next_mins, next_maxs, 0, 2)

        return next_crit

    def crit_line(self, start, start_at_bif):
        nexts = []
        critline = CritLine()
        if not start_at_bif:
            critline.crit_list.append(start)
        else:
            critline.bifurcation_point = start


        for i in range(2):
            temp = self.fff_step(start, i)
            if temp:
                nexts.append([temp[0], i])

        while len(nexts) > 0:
            current = nexts[0]
            #print(current)
            nexts.remove(current)
            if current[1] == 0:
                critline.crit_list.append(current[0])
            else:
                critline.crit_list.insert(0, current[0])
            in_range = True
            for i in range(self.space_dimensions+1):
                if current[0][i] < self.MIN or current[0][i] > self.MAX:
                    in_range = False

            if in_range:
                temp = self.fff_step(current[0], current[1])
                if temp:
                    nexts.append([temp[0], current[1]])

        return critline


vf = VectorField(f_2p2d_circle, 2, 2)
m_vf = VectorField([f_2p2d_circle_z, f_2p2d_circle_x, f_2p2d_circle_y], 1, 3)

bif_line = m_vf.crit_line([0.0, 1.0, 0.0, 0.0], True)
bifurcations = bif_line.crit_list.copy()

print(len(bifurcations))
for i in range(len(bifurcations)):
    print(i)
    crit_line = vf.crit_line(bifurcations[i], True)
    bif_line.lower_order_critlines.append(crit_line)

points = vtkPoints()
vertices = vtkCellArray()
polydata = vtkPolyData()
cells = vtkCellArray()
colors = vtkUnsignedCharArray()

colors.SetNumberOfComponents(4)
colors.SetNumberOfTuples(len(bif_line.lower_order_critlines)+1)
id = 0

rgba = [0, 255, 0, 255]
colors.InsertTuple(0, rgba)
bif_poly_line = vtkPolyLine()
bif_poly_line.GetPointIds().SetNumberOfIds(len(bifurcations))
for i in range(len(bif_line.crit_list)):
    points.InsertNextPoint([bif_line.crit_list[i][0], bif_line.crit_list[i][1], bif_line.crit_list[i][2]])
    bif_poly_line.GetPointIds().SetId(i, id)
    id = id + 1
cells.InsertNextCell(bif_poly_line)
for i in range(len(bif_line.lower_order_critlines)):
    rgba = [255, 0, 0, 255]
    colors.InsertTuple(i+1, rgba)
    crit_poly_line = vtkPolyLine()
    crit_poly_line.GetPointIds().SetNumberOfIds(len(bif_line.lower_order_critlines[i].crit_list))
    for j in range(len(bif_line.lower_order_critlines[i].crit_list)):
        points.InsertNextPoint([bif_line.lower_order_critlines[i].crit_list[j][0], bif_line.lower_order_critlines[i].crit_list[j][1], bif_line.lower_order_critlines[i].crit_list[j][2]])
        crit_poly_line.GetPointIds().SetId(j, id)
        id = id + 1
    cells.InsertNextCell(crit_poly_line)

polydata.SetLines(cells)
polydata.SetPoints(points)
polydata.GetCellData().SetScalars(colors);


#    colors = vtkUnsignedCharArray()
#    colors.SetNumberOfComponents(4)
#    colors.SetNumberOfTuples(len(crits)+len(bifurcations))
#
#    for i in range(len(crits)):
#        rgba = [255, 0, 0, 255]
#        colors.InsertTuple(i, rgba)
#        p = [crits[i][0], crits[i][1], crits[i][2]]
#        pid = points.InsertNextPoint(p)
#        vertices.InsertNextCell(1)
#        vertices.InsertCellPoint(pid)
#
#    for i in range(len(bifurcations)):
#        rgba = [0, 255, 0, 255]
#        colors.InsertTuple(i+len(crits), rgba)
#        p = [bifurcations[i][0], bifurcations[i][1], bifurcations[i][2]]
#        pid = points.InsertNextPoint(p)
#        vertices.InsertNextCell(1)
#        vertices.InsertCellPoint(pid)
#
#
#    polydata.SetPoints(points)
#    polydata.SetVerts(vertices)
#    polydata.GetCellData().SetScalars(colors);



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

#8 Seiten + Refferenzen
#Sehr weit vorne anfangen; Kritische Punkte, Parameterabhängige Vektorfelder, FFF, Bifurkationen; Man muss alles durch das Paper verstehen; Stromlinie = Integralkurve des stationären Feldes;
#Introduction
#Related Works
#Fundamentals
#Method
#Konzept
#Diskretisierung
#1/4 Seite Implementierung
#Results

#Feld Extraktion nochmal genauer angucken

#Parametrisierung aufschreiben (Länge entlang der einen Richtung, Länge entlang der zweiten Kurve)

#Newton Iteration nochmal neu probieren
#Stable Feature Flow Field
#Philipp entlang hoher Mannigfaltigkeiten mit Wedge Product

#Höhere Bifurcation kann durch Vorzeichenwechsel erkannt werden, wenn die Schrittweise über 1 ist

#Bei welchen Ids fehlen ide Linien: An die Stellen Debuggen

#Folien beschreiben was ich gemacht habe
#Work Packages

#mark.ladd@dkfz-heidelberg.de
#dachsbacher@kit.edu
# TAC Meeting: Mail an Mark Ladd (DKFZ) schreiben; Karsten Dachsbacher (HIDSS4Health TAC) einladen; Doodle aufsetzen; 25.7-29.7; Wenn es nicht geht, alternativen anbieten

# TODO Promotionsunterlagen
# TODO Phd Student Gitlab Project anlegen
# TODO Paper Gitlab Project anlegen
# TODO HGS anmelden

# TODO Move to Bifurcation from Start
# TODO Template runterladen
# TODO Rand Bifurcationen berechnen
# TODO Flächen rendern
# TODO Fehlende Linien fixen
# TODO Merge CritLines

#TAC Meeting: 2-3 Wochen; Mark Ladd, Karsten Dachsbacher (KIT); Filip Sadlo
#Orthogonal: Dependent Vectors Operator; Wedge Product
#Dependent Vectors Flow Field
#Periodische Orbits finden? Bifurcation Gegenstück dazu? Mehr als zwei Parameter; Boundary Switch Curves? (Theisel, Weinkauf)(Bifurkationsstrukturen davon); Analyse schneller als Generierung der Daten; Großteil der hochdimensionalen Daten vermutlich analytisch


#Erhöhtes Delta + Min Recursion bei Step = volles Kreis, aber nur halbe Kugel
# Wenn Kreis voll, sucht er von vorne und hinten die critline, was zu gedoppelten Punkten führt

#FFF Initial:               http://vc.cs.ovgu.de/files/publications/Archive/Theisel_2003_VisSym.pdf
#FFF korrekt(er):           http://vc.cs.ovgu.de/index.php?article_id=3&clang=0&bibtex_key=Theisel_2003_VisSym
#Calculate Orthogonal:      https://www.geometrictools.com/Documentation/OrthonormalSets.pdf
#FFF Allgemein:             Sign nur bei ungerader Anzahl an Dimensionen (entweder über Determinante oder Gram-Schmidts)