import numpy as np
import pygame
import cython
import random
import colorsys

cdef int N = 128
cdef int iterations = 16

cdef inline int IX(int x_coord, int y_coords):
        return x_coord + y_coords*N

cdef inline int denorm(tup):
    return (tup[0]*255, tup[1]*255, tup[2]*255)

cdef float visc, diff, dt, x , y, d, vx, vy, angle
cdef float [:] Vx, Vy, Vx0, Vy0, s, density
cdef int i, j, k, cx, cy
cdef int scale = 4

cdef float i0, i1, j0, j1, dtx, dty, s0, s1, t0, t1, tmp1, tmp2, Nfloat, ifloat, jfloat


class Fluid():
    def __init__(self, dt, diffusion, viscosity):
        global N
        self.size = N
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity

        self.s = np.zeros((self.size*self.size), dtype=np.float32)
        self.density = np.zeros((self.size*self.size), dtype=np.float32)


        self.Vx = np.zeros((self.size*self.size), dtype=np.float32)
        self.Vy = np.zeros((self.size*self.size), dtype=np.float32)

        self.Vx0 = np.zeros((self.size*self.size), dtype=np.float32)
        self.Vy0 = np.zeros((self.size*self.size), dtype=np.float32)


    def addDensity(self, int x_coord, int y_coord, float amount):
        self.density[IX(x_coord, y_coord)] += amount

    def addVelocity(self, int x_coord, int y_coord, float addVelX, float addVelY):
        cdef index = IX(x_coord, y_coord)

        self.Vx[index] += addVelX
        self.Vy[index] += addVelY

    def step(self):
        visc     = self.visc;
        diff     = self.diff;
        dt       = self.dt;
        Vx      = self.Vx;
        Vy      = self.Vy;
        Vx0     = self.Vx0;
        Vy0     = self.Vy0;
        s       = self.s;
        density = self.density;

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
                d = self.density[IX(i, j)]
                pygame.draw.rect(screen, denorm(colorsys.hsv_to_rgb(((d + 50) % 360) / 360,200 / 200, np.clip(d// 3, 0, 100)/ 100)), (x, y, scale, scale))

    def renderV(self, screen):

        for i in range(N):
            for j in range(N):
                x = i * scale
                y = j * scale
                vx = self.Vx[IX(i, j)];
                vy = self.Vy[IX(i, j)];

                if (not (abs(vx) < 0.1 and abs(vy) <= 0.1)):
                    pygame.draw.line(screen, x, y, x+vx*scale, y+vy*scale, 1)

    def fadeD(self, screen):
        for i in range(len(self.density)):
            d = self.density[i]
            self.density[i] = np.clip(d-0.02, 0, 255)

def diffuse (int b, float [:] x_array, float [:] x0_array, float diff, float dt):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x_array, x0_array, a, 1 + 6 * a)


def lin_solve(int b, float [:] x_array, float [:] x0_array, float a, float c):
    cdef float cRecip = 1.0 / c
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


    set_bnd(0, div_array);
    set_bnd(0, p_array);
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
            i0 = np.floor(x)
            i1 = i0 + 1.0
            if(y < 0.5): y = 0.5
            if(y > Nfloat + 0.5): y = Nfloat + 0.5
            j0 = np.floor(y)
            j1 = j0 + 1.0

            s1 = x - i0;
            s0 = 1.0 - s1;
            t1 = y - j0;
            t0 = 1.0 - t1;

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


pygame.init()
screen = pygame.display.set_mode((N * scale, N * scale))
done = False

fluid = Fluid(0.2, 0, 0.0000001)

color = 0
colors = [(255, 255, 255), (0,0,0)]

while not done:
    screen.fill((0,0,0))
    cx = int(0.5*N)
    cy = int(0.5*N)
    for i in range(-1, 2):
        for j in range(-1, 2):
            fluid.addDensity(cx+i, cy+j, random.randint(50, 150))

    for i in range(-1, 2):
        angle = (random.random() * 2 * np.pi) - np.pi
        fluid.addVelocity(cx, cy, np.cos(angle) * 0.2, np.cos(angle) * 0.2)


    fluid.step()
    fluid.renderD(screen)
    # fluid.renderV(screen)
    # fluid.fadeD(screen)

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    done = True

    pygame.display.flip()
pygame.quit()
