import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


# Input parameters

File_path = 'bumpgrid.dat.txt'   # grid file path

gamma = 1.4                     # ratio of specific heats

# Freestream conditions
Mach    = 1.4 
P_inf   = 1.0*1e+3
T_inf   = 300
rho_inf = P_inf/(T_inf*287)
a_inf   = np.sqrt(gamma*P_inf/rho_inf)  # a_inf = sqrt(gamma p / rho) = 1
u_inf   = Mach*a_inf                   
v_inf   = 0.0       

CFL     = 0.8
tol     = 1e-6
niter   = 5000


# Grid reading 

data = np.loadtxt(File_path)
data = np.array(data)

[imx, jmx] = data[0, :]
imx = int(imx)
jmx = int(jmx)

X = np.zeros([imx, jmx])
Y = np.zeros([imx, jmx])

for j in range(jmx):
    for i in range(imx):
        X[i, j] = data[j * imx + i + 1, 0]
        Y[i, j] = data[j * imx + i + 1, 1]


# Geometry: face normals & cell volumes 

xnx = np.zeros([imx + 1, jmx + 1])
xny = np.zeros([imx + 1, jmx + 1])

# i-direction faces
for j in range(1, jmx):
    for i in range(imx):
        xnx[i, j] = Y[i, j] - Y[i, j - 1]
        xny[i, j] = -(X[i, j] - X[i, j - 1])

ynx = np.zeros([imx + 1, jmx + 1])
yny = np.zeros([imx + 1, jmx + 1])

# j-direction faces
for j in range(jmx):
    for i in range(1, imx):
        ynx[i, j] = Y[i - 1, j] - Y[i, j]
        yny[i, j] = -(X[i - 1, j] - X[i, j])

vol = np.zeros([imx + 1, jmx + 1])

for j in range(1, jmx):
    for i in range(1, imx):
        x_1, y_1 = X[i - 1, j - 1], Y[i - 1, j - 1]
        x_2, y_2 = X[i, j - 1], Y[i, j - 1]
        x_3, y_3 = X[i, j], Y[i, j]
        x_4, y_4 = X[i - 1, j], Y[i - 1, j]
        vol[i, j] = 0.5 * (
            abs((x_2 - x_1) * (y_4 - y_2) - (y_2 - y_1) * (x_4 - x_1)) +
            abs((x_4 - x_3) * (y_2 - y_3) - (y_4 - y_3) * (x_2 - x_3))
        )


# Helper functions

def conservative_to_primitive(rho, rhou, rhov, rhoE, gamma):
    """Convert conservative to primitive variables."""
    u = rhou / rho
    v = rhov / rho
    vel2 = u**2 + v**2
    P = (gamma - 1.0) * (rhoE - 0.5 * rho * vel2)
    return u, v, P

def primitive_to_conservative(rho, u, v, P, gamma):
    """Convert primitive to conservative variables."""
    vel2 = u**2 + v**2
    E = P / ((gamma - 1.0) * rho) + 0.5 * vel2
    rhoE = rho * E
    return rho, rho * u, rho * v, rhoE

def roe_dissipation_flux(qL, qR, nx, ny, gamma):
    """
    Compute Roe dissipation vector d (already multiplied by face area A)
    qL, qR: conservative vectors [rho, rhou, rhov, rhoE]
    nx, ny: face normal components (not normalized)
    """
    eps = 1e-14

    A = math.sqrt(nx**2 + ny**2) + eps
    nx_hat = nx/A
    ny_hat = ny/A

    # Left state
    rhol = qL[0]
    ul = qL[1]/rhol
    vl = qL[2]/rhol
    Pl = (gamma - 1.0) * (qL[3] - 0.5 * rhol * (ul**2 + vl**2))
    Hl = (qL[3] + Pl) / rhol

    # Right state
    rhor = qR[0]
    ur = qR[1] / rhor
    vr = qR[2] / rhor
    Pr = (gamma - 1.0) * (qR[3] - 0.5 * rhor * (ur**2 + vr**2))
    Hr = (qR[3] + Pr) / rhor

    # Normal velocities
    vnl = ul * nx_hat + vl * ny_hat
    vnr = ur * nx_hat + vr * ny_hat

    # Roe averages
    r = math.sqrt(max(rhor / max(rhol, eps), eps))
    rhoR = math.sqrt(max(rhol * rhor, eps))
    u_r = (ul + r * ur) / (1.0 + r)
    vR = (vl + r * vr) / (1.0 + r)
    H0R = (Hl + r * Hr) / (1.0 + r)

    aR2 = (gamma - 1.0) * (H0R - 0.5 * (u_r * u_r + vR * vR))
    aR2 = max(aR2, eps)
    aR = math.sqrt(aR2)

    vnR = u_r*nx_hat + vR*ny_hat

    # Jumps
    d_rho = rhor - rhol
    d_P   = Pr - Pl
    d_vn  = vnr - vnl
    d_u   = ur - ul
    d_v   = vr - vl

    # alpha's
    alpha1 = A * abs(vnR) * (d_rho - d_P / aR2)
    alpha2 = A / (2.0 * aR2) * abs(vnR + aR) * (d_P + rhoR * aR * d_vn)
    alpha3 = A / (2.0 * aR2) * abs(vnR - aR) * (d_P - rhoR * aR * d_vn)

    alpha4 = alpha1 + alpha2 + alpha3
    alpha5 = aR * (alpha2 - alpha3)
    alpha6 = A * abs(vnR) * (rhoR * d_u - nx_hat * rhoR * d_vn)
    alpha7 = A * abs(vnR) * (rhoR * d_v - ny_hat * rhoR * d_vn)

    d1 = alpha4
    d2 = u_r * alpha4 + nx_hat * alpha5 + alpha6
    d3 = vR * alpha4 + ny_hat * alpha5 + alpha7
    d4 = (H0R * alpha4 + vnR * alpha5 + u_r * alpha6 + vR * alpha7 - aR2 * alpha1 / (gamma - 1.0))

    return np.array([d1, d2, d3, d4])

def central_flux(q, nx, ny, gamma):
    """
    Compute physical flux (Euler) dotted with unit normal,
    multiplied by area A (so this returns integrated face flux).
    q: conservative vector
    """
    tiny = 1e-14
    A = math.sqrt(nx * nx + ny * ny) + tiny
    nx_hat = nx / A
    ny_hat = ny / A

    rho = q[0]
    u   = q[1] / rho
    v   = q[2] / rho
    P   = (gamma - 1.0) * (q[3] - 0.5 * rho * (u**2 + v**2))

    vn = u * nx_hat + v * ny_hat

    F1 = rho * vn
    F2 = rho * u * vn + P * nx_hat
    F3 = rho * v * vn + P * ny_hat
    F4 = (q[3] + P) * vn

    return A * np.array([F1, F2, F3, F4])

def save_solution(iter_num, rho, rhou, rhov, rhoE, gamma, X, Y):
    """Save primitive solution (rho, u, v, p) at nodes to CSV."""
    imx, jmx = X.shape
    u_node = np.zeros((imx, jmx))
    v_node = np.zeros((imx, jmx))
    p_node = np.zeros((imx, jmx))
    rho_node = np.zeros((imx, jmx))

    # cell-centered values are at [1:imx,1:jmx] in arrays of size (imx+1,jmx+1)
    for j in range(jmx):
        for i in range(imx):
            # average of surrounding 4 cells
            r_c = 0.25 * (rho[i, j] + rho[i + 1, j] +
                          rho[i, j + 1] + rho[i + 1, j + 1])
            ru_c = 0.25 * (rhou[i, j] + rhou[i + 1, j] +
                           rhou[i, j + 1] + rhou[i + 1, j + 1])
            rv_c = 0.25 * (rhov[i, j] + rhov[i + 1, j] +
                           rhov[i, j + 1] + rhov[i + 1, j + 1])
            rE_c = 0.25 * (rhoE[i, j] + rhoE[i + 1, j] +
                           rhoE[i, j + 1] + rhoE[i + 1, j + 1])

            uc, vc, pc = conservative_to_primitive(r_c, ru_c, rv_c, rE_c, gamma)
            u_node[i, j] = uc
            v_node[i, j] = vc
            p_node[i, j] = pc
            rho_node[i, j] = r_c

    output_data = []
    for j in range(jmx):
        for i in range(imx):
            output_data.append([
                X[i, j], Y[i, j],
                rho_node[i, j],
                u_node[i, j],
                v_node[i, j],
                p_node[i, j]
            ])

    df = pd.DataFrame(output_data,
                      columns=['x', 'y', 'rho', 'u', 'v', 'p'])
    filename = f"solution_roe_iter_{Mach}_{iter_num}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved solution to {filename}")


# Initialize conservative variables

# Arrays are (imx+1, jmx+1) to include ghost layers
rho  = np.full((imx + 1, jmx + 1), rho_inf)
u    = np.full((imx + 1, jmx + 1), u_inf)
v    = np.full((imx + 1, jmx + 1), v_inf)
P    = np.full((imx + 1, jmx + 1), P_inf)

rho, rhou, rhov, rhoE = primitive_to_conservative(rho, u, v, P, gamma)

res = np.zeros((4, imx + 1, jmx + 1))
delt = np.zeros((imx + 1, jmx + 1))

residual_norms = []

# Scaling for residuals (based on freestream)
scale = np.zeros(4)
a_inf = math.sqrt(gamma * P_inf / rho_inf)
scale[0] = rho_inf * u_inf       # mass
scale[1] = rho_inf * u_inf**2    # x-mom
scale[2] = rho_inf * u_inf**2    # y-mom
scale[3] = rho_inf * u_inf * ((gamma/(gamma - 1)) * P_inf/rho_inf + 0.5*u_inf**2)   # energy

eps = 1e-14

def compute_residual(rho, rhou, rhov, rhoE):
    """
    Given conservative variables (with ghost cells),
    apply BCs, compute Roe fluxes, and return residual array res
    and its (scaled) L2 norm.
    """
    # primitives
    u = rhou / rho
    v = rhov / rho
    vel2 = u * u + v * v
    p = (gamma - 1.0) * (rhoE - 0.5 * rho * vel2)

    if Mach > 1:
        rho[0, :] = rho_inf
        u[0, :] = u_inf
        v[0, :] = v_inf
        P[0, :] = P_inf
    else:
        rho[0, :] = rho_inf
        u[0, :] = u_inf
        v[0, :] = v_inf
        P[0, :] = P[1, :]

    
    rho[0, :], rhou[0, :], rhov[0, :], rhoE[0, :] = primitive_to_conservative(rho[0, :], u[0, :], v[0, :], p[0, :], gamma)

    # Outflow at i = imx 

    rho[imx, :]  = rho[imx - 1, :]
    u[imx, :]    = u[imx - 1, :]
    v[imx, :]    = v[imx - 1, :]
    P[imx, :]    = P[imx - 1, :]

    rho[imx, :], rhou[imx, :], rhov[imx, :], rhoE[imx, :] = primitive_to_conservative(rho[imx, :], u[imx, :], v[imx, :], P[imx, :], gamma)


    # Recompute primitives after outlet BC
    u = rhou / rho
    v = rhov / rho
    vel2 = u * u + v * v
    p = (gamma - 1.0) * (rhoE - 0.5 * rho * vel2)

    # Bottom wall (j=0 ghost, j=1 interior), wall

    j = 0
    for i in range(imx + 1):
        area = math.sqrt(ynx[i, j]**2 + yny[i, j]**2) + eps
        v_n = u[i, j+1]*ynx[i, j]/area + v[i, j+1]*yny[i, j]/area
        rho[i, j] = rho[i, j+1]
        u[i, j] = u[i, j+1] - 2*v_n*ynx[i, j]/area
        v[i, j] = v[i, j+1] - 2*v_n*yny[i, j]/area
        P[i, j] = P[i, j+1]
        
        rho[i, j], rhou[i, j], rhov[i, j], rhoE[i, j] = primitive_to_conservative(rho[i, j], u[i, j], v[i, j], P[i, j], gamma)

    # Top Wall (j=jmx ghost, j=jmx-1 interior)

    j = jmx
    for i in range(imx + 1):
        area = math.sqrt(ynx[i, j - 1]**2 + yny[i, j - 1]**2) + eps
        v_n = u[i, j - 1]*ynx[i, j - 1]/area + v[i, j - 1]*yny[i, j - 1]/area
        rho[i, j] = rho[i, j - 1]
        u[i, j] = u[i, j - 1] - 2*v_n*ynx[i, j - 1]/area
        v[i, j] = v[i, j - 1] - 2*v_n*yny[i, j - 1]/area
        P[i, j] = P[i, j - 1]
        
        rho[i, j], rhou[i, j], rhov[i, j], rhoE[i, j] = primitive_to_conservative(rho[i, j], u[i, j], v[i, j], P[i, j], gamma)


    # Fluxes with Roe dissipation 
    xflux = np.zeros((4, imx + 1, jmx + 1))
    yflux = np.zeros((4, imx + 1, jmx + 1))

    # i-face fluxes
    for j in range(1, jmx):
        for i in range(imx):
            qL = np.array([rho[i, j], rhou[i, j], rhov[i, j], rhoE[i, j]])
            qR = np.array([rho[i + 1, j], rhou[i + 1, j], rhov[i + 1, j], rhoE[i + 1, j]])

            nx = xnx[i, j]
            ny = xny[i, j]

            F_c = 0.5 * (central_flux(qL, nx, ny, gamma) +
                         central_flux(qR, nx, ny, gamma))
            d = roe_dissipation_flux(qL, qR, nx, ny, gamma)

            xflux[:, i, j] = F_c - 0.5 * d

    # j-face fluxes
    for j in range(jmx):
        for i in range(1, imx):
            qL = np.array([rho[i, j], rhou[i, j], rhov[i, j], rhoE[i, j]])
            qR = np.array([rho[i, j + 1], rhou[i, j + 1], rhov[i, j + 1], rhoE[i, j + 1]])

            nx = ynx[i, j]
            ny = yny[i, j]

            F_c = 0.5 * (central_flux(qL, nx, ny, gamma) +
                         central_flux(qR, nx, ny, gamma))
            d = roe_dissipation_flux(qL, qR, nx, ny, gamma)

            yflux[:, i, j] = F_c - 0.5 * d

    # Residual (finite-volume balance) 
    res_local = np.zeros((4, imx + 1, jmx + 1))
    resnorm = 0.0

    for j in range(1, jmx):
        for i in range(1, imx):
            res_local[:, i, j] = ((xflux[:, i, j] - xflux[:, i - 1, j]) +
                                  (yflux[:, i, j] - yflux[:, i, j - 1]))

            # residual norm (same scaling as before)
            for k in range(4):
                resnorm += (res_local[k, i, j] ** 2) / max(scale[k], eps)

    return res_local, resnorm


def compute_dt_over_vol(rho, rhou, rhov, rhoE):
    """
    Compute local dt/vol using the CFL formula based on primitive variables.
    """
    u = rhou / rho
    v = rhov / rho
    vel2 = u * u + v * v
    p = (gamma - 1.0) * (rhoE - 0.5 * rho * vel2)

    delt_local = np.zeros_like(rho)

    for j in range(1, jmx):
        for i in range(1, imx):
            # left face
            nx_l = xnx[i - 1, j]
            ny_l = xny[i - 1, j]
            A_l = math.sqrt(nx_l * nx_l + ny_l * ny_l) + eps
            u_l = u[i, j]
            v_l = v[i, j]
            press = p[i, j]
            speed_sound_loc = math.sqrt(max(gamma * press / rho[i, j], 0))
            vn_l = (u_l * nx_l + v_l * ny_l) / A_l
            lam_l = A_l * (abs(vn_l) + speed_sound_loc)

            # right face
            nx_r = xnx[i, j]
            ny_r = xny[i, j]
            A_r = math.sqrt(nx_r * nx_r + ny_r * ny_r) + eps
            vn_r = (u[i, j] * nx_r + v[i, j] * ny_r) / A_r
            lam_r = A_r * (abs(vn_r) + speed_sound_loc)

            # bottom face
            nx_b = ynx[i, j - 1]
            ny_b = yny[i, j - 1]
            A_b = math.sqrt(nx_b * nx_b + ny_b * ny_b) + eps
            vn_b = (u[i, j] * nx_b + v[i, j] * ny_b) / A_b
            lam_b = A_b * (abs(vn_b) + speed_sound_loc)

            # top face
            nx_t = ynx[i, j]
            ny_t = yny[i, j]
            A_t = math.sqrt(nx_t * nx_t + ny_t * ny_t) + eps
            vn_t = (u[i, j] * nx_t + v[i, j] * ny_t) / A_t
            lam_t = A_t * (abs(vn_t) + speed_sound_loc)

            lam_sum = lam_l + lam_r + lam_b + lam_t + eps
            delt_local[i, j] = CFL * 2.0 * vol[i, j] / lam_sum

    # Convert to dt_over_vol for convenience (For RK4 update)

    dt_over_vol = np.zeros_like(rho)
    for j in range(1, jmx):
        for i in range(1, imx):
            dt_over_vol[i, j] = delt_local[i, j] / max(vol[i, j], eps)

    return dt_over_vol



# Main iteration loop (RK4 in pseudo-time)

for iter in range(niter):

    # compute local dt/vol once from current state
    dt_over_vol = compute_dt_over_vol(rho, rhou, rhov, rhoE)

    # -Stage 1 
    # work on copies because compute_residual modifies arrays via BCs
    rho1  = rho.copy()
    rhou1 = rhou.copy()
    rhov1 = rhov.copy()
    rhoE1 = rhoE.copy()

    res1, resnorm = compute_residual(rho1, rhou1, rhov1, rhoE1)

    if iter == 0:
        resnorm0 = resnorm if resnorm > 0 else 1.0

    resnorm_rel = resnorm / resnorm0
    residual_norms.append(resnorm_rel)

    if iter % 10 == 0:
        print(f"Iteration {iter}: Residual = {resnorm_rel:.6e}")

    if resnorm_rel < tol:
        print(f"Converged at iteration {iter} with residual {resnorm_rel:.6e}")
        break

    # Stage 2 
    rho2  = rho.copy()
    rhou2 = rhou.copy()
    rhov2 = rhov.copy()
    rhoE2 = rhoE.copy()

    for j in range(1, jmx):
        for i in range(1, imx):
            rho2[i, j]  -= 0.5 * dt_over_vol[i, j] * res1[0, i, j]
            rhou2[i, j] -= 0.5 * dt_over_vol[i, j] * res1[1, i, j]
            rhov2[i, j] -= 0.5 * dt_over_vol[i, j] * res1[2, i, j]
            rhoE2[i, j] -= 0.5 * dt_over_vol[i, j] * res1[3, i, j]

    rho2c  = rho2.copy()
    rhou2c = rhou2.copy()
    rhov2c = rhov2.copy()
    rhoE2c = rhoE2.copy()
    res2, _ = compute_residual(rho2c, rhou2c, rhov2c, rhoE2c)

    # Stage 3
    rho3  = rho.copy()
    rhou3 = rhou.copy()
    rhov3 = rhov.copy()
    rhoE3 = rhoE.copy()

    for j in range(1, jmx):
        for i in range(1, imx):
            rho3[i, j]  -= 0.5 * dt_over_vol[i, j] * res2[0, i, j]
            rhou3[i, j] -= 0.5 * dt_over_vol[i, j] * res2[1, i, j]
            rhov3[i, j] -= 0.5 * dt_over_vol[i, j] * res2[2, i, j]
            rhoE3[i, j] -= 0.5 * dt_over_vol[i, j] * res2[3, i, j]

    rho3c  = rho3.copy()
    rhou3c = rhou3.copy()
    rhov3c = rhov3.copy()
    rhoE3c = rhoE3.copy()
    res3, _ = compute_residual(rho3c, rhou3c, rhov3c, rhoE3c)

    # Stage 4
    rho4  = rho.copy()
    rhou4 = rhou.copy()
    rhov4 = rhov.copy()
    rhoE4 = rhoE.copy()

    for j in range(1, jmx):
        for i in range(1, imx):
            rho4[i, j]  -= dt_over_vol[i, j] * res3[0, i, j]
            rhou4[i, j] -= dt_over_vol[i, j] * res3[1, i, j]
            rhov4[i, j] -= dt_over_vol[i, j] * res3[2, i, j]
            rhoE4[i, j] -= dt_over_vol[i, j] * res3[3, i, j]

    rho4c  = rho4.copy()
    rhou4c = rhou4.copy()
    rhov4c = rhov4.copy()
    rhoE4c = rhoE4.copy()
    res4, _ = compute_residual(rho4c, rhou4c, rhov4c, rhoE4c)

    # Final RK4 combination
    for j in range(1, jmx):
        for i in range(1, imx):
            factor = dt_over_vol[i, j] / 6.0
            rho[i, j]  -= factor * (res1[0, i, j] + 2.0 * res2[0, i, j] +
                                    2.0 * res3[0, i, j] + res4[0, i, j])
            rhou[i, j] -= factor * (res1[1, i, j] + 2.0 * res2[1, i, j] +
                                    2.0 * res3[1, i, j] + res4[1, i, j])
            rhov[i, j] -= factor * (res1[2, i, j] + 2.0 * res2[2, i, j] +
                                    2.0 * res3[2, i, j] + res4[2, i, j])
            rhoE[i, j] -= factor * (res1[3, i, j] + 2.0 * res2[3, i, j] +
                                    2.0 * res3[3, i, j] + res4[3, i, j])

    # BCs will be re-applied at the start of the next iteration inside compute_residual


# Final save and plots
save_solution(iter, rho, rhou, rhov, rhoE, gamma, X, Y)

# Convergence history
plt.figure(figsize=(8, 5))
plt.semilogy(residual_norms)
plt.xlabel('Iteration')
plt.ylabel('Normalized Residual')
plt.title(f'Convergence History (Roe scheme) (Mach = {Mach})')
plt.grid(True, alpha=0.3)
plt.savefig(f'convergence_roe_Mach_{Mach}.png', dpi=300, bbox_inches='tight')
plt.show()

# Simple contour plots at nodes

u_node = np.zeros((imx, jmx))
v_node = np.zeros((imx, jmx))
p_node = np.zeros((imx, jmx))
Mach_node = np.zeros((imx,jmx))
rho_node = np.zeros((imx, jmx))

for j in range(jmx):
    for i in range(imx):
        r_c = 0.25 * (rho[i, j] + rho[i + 1, j] +
                      rho[i, j + 1] + rho[i + 1, j + 1])
        ru_c = 0.25 * (rhou[i, j] + rhou[i + 1, j] +
                       rhou[i, j + 1] + rhou[i + 1, j + 1])
        rv_c = 0.25 * (rhov[i, j] + rhov[i + 1, j] +
                       rhov[i, j + 1] + rhov[i + 1, j + 1])
        rE_c = 0.25 * (rhoE[i, j] + rhoE[i + 1, j] +
                       rhoE[i, j + 1] + rhoE[i + 1, j + 1])

        uc, vc, pc = conservative_to_primitive(r_c, ru_c, rv_c, rE_c, gamma)
        rho_node[i,j] = r_c 
        u_node[i, j] = uc
        v_node[i, j] = vc
        p_node[i, j] = pc

        a_node = np.sqrt(gamma*pc/r_c)
        Mach_node[i,j] = math.sqrt(uc**2 + vc**2)/a_node

# Pressure contour
plt.figure()
plt.contourf(X, Y, p_node, levels=50)
plt.colorbar()
plt.title(f'Pressure (Roe Scheme) (Mach = {Mach}) ')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f'Pressure_Contour_Roe_Mach_{Mach}.png', dpi=300, bbox_inches='tight')

# Velocity magnitude
vel_mag = np.sqrt(u_node**2 + v_node**2)
plt.figure()
plt.contourf(X, Y, vel_mag, levels=50)
plt.colorbar()
plt.title(f'Velocity Magnitude (Roe Scheme) (Mach = {Mach})')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f'Velocity_Magnitude_Roe_{Mach}.png.png', dpi=300, bbox_inches='tight')

# Velocity vectors
plt.figure()
skip = max(1, imx // 40)
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
           u_node[::skip, ::skip], v_node[::skip, ::skip])
plt.title(f'Velocity Vectors (Roe Scheme) (Mach = {Mach})')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(f'Velocity_Vectors_Roe_{Mach}.png.png', dpi=300, bbox_inches='tight')
plt.show()

# Mach number plot
plt.figure()
plt.figure()
plt.contourf(X, Y, Mach_node, levels=50)
plt.colorbar()
plt.title(f'Mach Numbers (Roe) (Mach = {Mach})')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(f'Mach_Roe_Mach_{Mach}.png', dpi=300, bbox_inches='tight')
plt.show()


# Numerical Schlieren from density

drdx = np.zeros_like(rho_node)
drdy = np.zeros_like(rho_node)

# central differences in physical space (x, y)
tiny = 1e-12
for j in range(1, jmx-1):
    for i in range(1, imx-1):
        dx = X[i+1, j] - X[i-1, j]
        dy = Y[i, j+1] - Y[i, j-1]

        if abs(dx) < tiny:
            dx = tiny
        if abs(dy) < tiny:
            dy = tiny

        drdx[i, j] = (rho_node[i+1, j] - rho_node[i-1, j]) / dx
        drdy[i, j] = (rho_node[i, j+1] - rho_node[i, j-1]) / dy

# gradient magnitude
grad_rho = np.sqrt(drdx**2 + drdy**2)

# avoid division by zero
max_grad = np.max(grad_rho)
if max_grad < tiny:
    max_grad = tiny

grad_rho_norm = grad_rho / max_grad

# Schlieren variable
kappa = 5.0   # controls contrast
alpha = 0.5   # controls how sharp the structures look
phi = np.exp(-kappa * grad_rho_norm**alpha)

# -------------
# Plot Schlieren
# -------------
plt.figure()
plt.contourf(X, Y, phi, levels=100, cmap='gray')
plt.gca().set_aspect('equal', 'box')
plt.title(f'Numerical Schlieren (|∇ρ|) – Mach = {Mach}')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Schlieren intensity')
plt.tight_layout()
plt.savefig(f'schlieren_Roe_Mach_{Mach}.png', dpi=300, bbox_inches='tight')
plt.show()