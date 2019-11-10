import numpy as np
import pygame
import cython
import random
import math
from noise import pnoise1
from pygame.locals import *

cdef int N = 256
cdef int iterations = 4
cdef int scale = 2
cdef float t = 0

cdef float visc, diff, dt, x , y, d, vx, vy, angle
cdef float [:] Vx, Vy, Vx0, Vy0, s, density
cdef int i, j, k, cx, cy, index
cdef float i0, i1, j0, j1, dtx, dty, s0, s1, t0, t1, tmp1, tmp2, Nfloat, ifloat, jfloat
cdef float cRecip

cdef inline int IX(int x_coord, int y_coord):
        return x_coord + y_coord*N


cdef class Fluid:
    cdef int size
    cdef float dt
    cdef float diff
    cdef float visc
    cdef float [:] s
    cdef float [:] density
    cdef float [:] Vx
    cdef float [:] Vy
    cdef float [:] Vx0
    cdef float [:] Vy0

    def __init__(self, dt, diffusion, viscosity):
        self.size = N
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity

        self.s = np.zeros(N*N, dtype=np.float32)
        self.density = np.zeros(N*N, dtype=np.float32)


        self.Vx = np.zeros(N*N, dtype=np.float32)
        self.Vy = np.zeros(N*N, dtype=np.float32)

        self.Vx0 = np.zeros(N*N, dtype=np.float32)
        self.Vy0 = np.zeros(N*N, dtype=np.float32)


    def addDensity(self, int x_coord, int y_coord, float amount):
        self.density[IX(x_coord, y_coord)] += amount

    def addVelocity(self, int x_coord, int y_coord, float addVelX, float addVelY):
        index = IX(x_coord, y_coord)

        self.Vx[index] += addVelX
        self.Vy[index] += addVelY

    def step(self):
        visc     = self.visc
        diff     = self.diff
        dt       = self.dt
        Vx      = self.Vx
        Vy      = self.Vy
        Vx0     = self.Vx0
        Vy0     = self.Vy0
        s       = self.s
        density = self.density

        diffuse(1, Vx0, Vx, visc, dt)
        diffuse(2, Vy0, Vy, visc, dt)

        project(Vx0, Vy0, Vx, Vy)

        advect(1, Vx, Vx0, Vx0, Vy0, dt)
        advect(2, Vy, Vy0, Vx0, Vy0, dt)

        project(Vx, Vy, Vx0, Vy0)

        diffuse(0, s, density, diff, dt)
        advect(0, density, s, Vx, Vy, dt)

    def renderD(self, screen):

        for i in range(N):
            for j in range(N):
                x = i * scale
                y = j * scale
                d = self.density[IX(i, j)] % 255
                pygame.draw.rect(screen, (d, d, d), (x, y, scale, scale))

    def renderV(self, screen):

        for i in range(N):
            for j in range(N):
                x = i * scale
                y = j * scale
                vx = self.Vx[IX(i, j)]
                vy = self.Vy[IX(i, j)]

                if (not (abs(vx) < 0.1 and abs(vy) <= 0.1)):
                    pygame.draw.line(screen, (0, 0, 0), (x, y), (x+vx*scale, y+vy*scale))

    def fadeD(self, screen):
        for i in range(len(self.density)):
            d = self.density[i]
            
            self.density[i] = np.clip(d-0.02, 0, 255)

def diffuse (int b, float [:] x_array, float [:] x0_array, float diff, float dt):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x_array, x0_array, a, 1 + 6 * a)


def lin_solve(int b, float [:] x_array, float [:] x0_array, float a, float c):
    cRecip = 1.0 / c
    for k in range(iterations):
        for j in range(1, N-1):
            for i in range(1, N-1):
                x_array[IX(i, j)] = (x0_array[IX(i, j)]
                                + a*(    x_array[IX(i+1, j  )]
                                +x_array[IX(i-1, j  )]
                                +x_array[IX(i  , j+1)]
                                +x_array[IX(i  , j-1)]
                       )) * cRecip



        set_bnd(b, x_array)

def project(float [:] velocX_array, float [:] velocY_array, float [:] p_array, float [:] div_array):
    for j in range(1, N-1):
        for i in range(1, N-1):
            div_array[IX(i, j)] = -0.5*(
                     velocX_array[IX(i+1, j  )]
                    -velocX_array[IX(i-1, j  )]
                    +velocY_array[IX(i  , j+1)]
                    -velocY_array[IX(i  , j-1)]
                )/N
            p_array[IX(i, j)] = 0


    set_bnd(0, div_array)
    set_bnd(0, p_array)
    lin_solve(0, p_array, div_array, 1, 6)


    for j in range(1, N-1):
        for i in range(1, N-1):
            velocX_array[IX(i, j)] -= 0.5 * (  p_array[IX(i+1, j)]
                                         -p_array[IX(i-1, j)]) * N
            velocY_array[IX(i, j)] -= 0.5 * (  p_array[IX(i, j+1)]
                                         -p_array[IX(i, j-1)]) * N

    set_bnd(1, velocX_array)
    set_bnd(2, velocY_array)

def advect(int b, float [:] d_array, float [:] d0_array,  float [:] velocX_array, float [:] velocY_array, float dt):
    dtx = dt * (N - 2)
    dty = dt * (N - 2)

    Nfloat = float(N)

    for j in range(1, N-1):
        for i in range(1, N-1):
            tmp1 = dtx * velocX_array[IX(i, j)]
            tmp2 = dty * velocY_array[IX(i, j)]
            x    = float(i) - tmp1
            y    = float(j) - tmp2

            if (x < 0.5): x = 0.5
            if(x > Nfloat + 0.5): x = Nfloat + 0.5
            i0 = math.floor(x)
            i1 = i0 + 1.0
            if(y < 0.5): y = 0.5
            if(y > Nfloat + 0.5): y = Nfloat + 0.5
            j0 = math.floor(y)
            j1 = j0 + 1.0

            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            i0i = int(i0)
            i1i = int(i1)
            j0i = int(j0)
            j1i = int(j1)

            d_array[IX(i, j)] = s0 * (( t0 *  d0_array[IX(i0i, j0i)]) +( t1 * d0_array[IX(i0i, j1i)])) + s1 * (( t0 * d0_array[IX(i1i, j0i)]) +( t1 * d0_array[IX(i1i, j1i)]))

    set_bnd(b, d_array)

def set_bnd(int b, float [:] x_array):
    for i in range(1, N-1):
        x_array[IX(i,  0 )] = -x_array[IX(i, 1  )] if b == 2 else x_array[IX(i, 1  )]
        x_array[IX(i, N-1)] = -x_array[IX(i, N-2)] if b == 2 else x_array[IX(i, N-2)]

    for j in range(1, N-1):
        x_array[IX(0  , j)] = -x_array[IX(1  , j)] if b == 1 else x_array[IX(1  , j)]
        x_array[IX(N-1, j)] = -x_array[IX(N-2, j)] if b == 1 else x_array[IX(N-2, j)]


    x_array[IX(0, 0)]     = 0.5 * (x_array[IX(1, 0)] + x_array[IX(0, 1)])
    x_array[IX(0, N-1)]   = 0.5 * (x_array[IX(1, N-1)] + x_array[IX(0, N-2)])
    x_array[IX(N-1, 0)]   = 0.5 * (x_array[IX(N-2, 0)] + x_array[IX(N-1, 1)])
    x_array[IX(N-1, N-1)] = 0.5 * (x_array[IX(N-2, N-1)] + x_array[IX(N-1, N-2)])


cdef int width = N*scale
cdef int height = N*scale
pygame.init()
flags = DOUBLEBUF
screen = pygame.display.set_mode(size=(width, height), flags=flags)
screen.set_alpha(None)
done = False

fluid = Fluid(1, 0, 0.0000001)

while not done:
    cx = int(0.5*width/scale)
    cy = int(0.5*height/scale)
    for i in range(-1, 2):
        for j in range(-1, 2):
            fluid.addDensity(cx+ i, cy+j, random.randint(50, 150))

    for i in range(2):
        angle = (pnoise1(t, 1) * 2 * math.pi)
        t += 0.01
        fluid.addVelocity(cx, cy, math.cos(angle) * 0.2, math.sin(angle) * 0.2)


    fluid.step()
    fluid.renderD(screen)
    # fluid.renderV(screen)
    # fluid.fadeD(screen)
    pygame.display.update()

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    done = True
    
pygame.quit()
