import numpy as np
import math

pi = math.pi

# 1) DONNÉES D’ENTRÉE

# Géométrie échangeur (cylindre interne / externe)
D_ext = 2.4     # [m]
D_int = 0.8     # [m]
L = 3.0        # [m]

# Tubes
Dtubeext = 0.010   # [m] diamètre extérieur tube
Dtubeint = 0.008   # [m] diamètre intérieur tube

# Maillage axial
Dx = 0.01          # [m] -> N = int(L/Dx)

# Débits massiques
qair = 7.9 #Donnée du sujet
qH2 = 0.31 #Donnée du sujet

# Températures d'entrée
Tair = 280.0       # [K] air entrant
TH2  = 40.0        # [K] H2 entrant au début de chaque étage

# Pression air
Pair = 1.0e5       # [Pa]

# Conductivité paroi Inconel
lambda_inconel = 12.0  # [W/m/K]

# 2) PROPRIÉTÉS

# defintion des paramètres de H2 en fonction de la température à la pression P

def lambda_air(T):
    """
    Calcul de la conductivité thermique de l'air en W/(m·K) en fonction de la température.
    T : Température en Kelvin (K)"""
    
    a, b, c = 2.528e-3, 7.487e-5, -1.562e-8  # coefficients spécifiques à l'air
    return (a + b * T + c * T**2)

def Cp_air(T):
    """
    Calcul de la chaleur spécifique de l'air en J/(kg·K) en fonction de la température.
    T : Température en Kelvin (K)
    P: Pression en Pa"""
    a, b, c, d = 1.003e3, 1.232e-1, -5.38e-5, 1.05e-8  # coefficients spécifiques à l'air
    return a + b * T + c * T**2 + d * T**3

def rho_air(T,P):
    """
    Calcul de la masse volumique de l'air en kg/m³ en fonction de la température.
    T : Température en Kelvin (K)
    P : Pression en Pascal (Pa) """
    R = 287.05  # Constante spécifique de l'air en J/(kg·K)
    return P / (R * T)

def visc_air(T):
    """
    Calcul de la viscosité dynamique de l'air en Pa·s en fonction de la température.
    T : Température en Kelvin (K)
    """
    mu_0 = 1.716e-5  # Viscosité de référence en Pa·s
    T_0 = 273.15  # Température de référence en K
    C = 111  # Constante de Sutherland en K
    return mu_0 * (T / T_0)**1.5 * (T_0 + C) / (T + C)

def Pr_air(T): #Nombre de Prandt
    return visc_air(T)*Cp_air(T)/lambda_air(T)

def alpha_air(T, P): #diffusivité thermique 
    return lambda_air(T)/(rho_air(T,P)/Cp_air(T))


# defintion des paramètres de H2 en fonction de la température à P=40 bars

def visc_H2(T):
    T = float(T)
    if T < 70.0:
        return 3.04167e-12*T**4 - 8.375e-10*T**3 + 8.68958e-8*T**2 - 4.01925e-6*T + 7.375e-5
    else:
        return 2.3239726028e-8*T + 2.2869863e-6


def Cp_H2(T):
    T = float(T)
    if T < 70.0:
        return -0.03924167*T**4 + 10.0305*T**3 - 942.835833*T**2 + 38327.85*T - 547104
    else:
        return -0.0216480226*T**3 + 7.5687203652*T**2 - 863.87635351*T + 45024.162274


def lambda_H2(T):
    T = float(T)
    if T < 70.0:
        return -5.236666666667e-7*T**3 + 0.0001321107143*T**2 - 0.0106742690476*T + 0.3486394285714
    else:
        return 4.1958333e-8*T**3 - 1.106875e-5*T**2 + 0.0013854917*T + 0.010585


def Pr_H2(T): #Nombre de Prandt
    return visc_H2(T)*Cp_H2(T)/lambda_H2(T)

def rho_H2(T): #Masse volumique
    return  -2.49415E-08*T**5 + 1.30799E-05*T**4 - 2.72782E-03*T**3 + 2.84577E-01*T**2 - 1.50762E+01*T + 3.43112E+02

def alpha_H2(T): #diffusivité thermique 
    return lambda_H2(T)/(rho_H2(T)*Cp_H2(T))

# 3) GEOMETRIE

def calcul_tubes_echangeur(d_tube):

    d_fictif = 4.0 * d_tube
    pas_radial = d_fictif

    N_couches_total = int(round((D_ext - D_int) / (2 * pas_radial)))

    # 1) Première zone
    D_premiere_couche = D_int + pas_radial
    perimetre_premiere = math.pi * D_premiere_couche
    N_tubes_couche1 = int(round(perimetre_premiere / d_fictif))

    # 2) Deuxième zone
    delta_tubes = 128
    N_tubes_couche2 = N_tubes_couche1 + delta_tubes

    perimetre_couche2 = N_tubes_couche2 * d_fictif
    D_couche2 = perimetre_couche2 / math.pi

    epaisseur_region2 = (D_ext - D_couche2) / 2
    N_couches_region2 = int(epaisseur_region2 / pas_radial)

    N_couches_region2 = max(0, min(N_couches_region2, N_couches_total))
    N_couches_region1 = N_couches_total - N_couches_region2

    total_tubes = (
        N_couches_region1 * N_tubes_couche1 +
        N_couches_region2 * N_tubes_couche2
    )

    return {
        "d_fictif": d_fictif,
        "N_couches_total": N_couches_total,
        "N_tubes_couche1": N_tubes_couche1,
        "N_tubes_couche2": N_tubes_couche2,
        "N_couches_region1": N_couches_region1,
        "N_couches_region2": N_couches_region2,
        "total_tubes": total_tubes
    }

# Calcul nombre de tubes via ta méthode
resultat = calcul_tubes_echangeur(Dtubeext)
Ntube = int(resultat["total_tubes"])

print("TOTAL TUBES DANS L'ÉCHANGEUR :", Ntube)

N_stages = Ntube

# Surfaces / sections d'écoulement à partir de Dint/Dext et Ntube
Rext = D_ext / 2.0
Rint = D_int / 2.0
A_annulus = pi * (Rext**2 - Rint**2)

A_tube_ext = pi * Dtubeext**2 / 4.0
A_tube_int = pi * Dtubeint**2 / 4.0

A_flow_H2  = Ntube * A_tube_int
A_flow_air = A_annulus - Ntube * A_tube_ext


print("A_flow_air =", A_flow_air, "m²")
print("A_flow_H2  =", A_flow_H2, "m²")
print("N_stages (étages) =", N_stages)

# 4) FONCTIONS : Re / h_air / h_H2

def reynolds(mdot, rho, mu, D_char, A_flow):
    U = mdot / (rho * A_flow)
    Re = rho * U * D_char / mu
    return Re, U

# AIR : convection forcée autour d’un cylindre
def hilpert_C_m(Re):
    if 0.4 <= Re < 4.0:
        C, m = 0.989, 0.33
    elif 4.0 <= Re < 40.0:
        C, m = 0.911, 0.385
    elif 40.0 <= Re < 4.0e3:
        C, m = 0.683, 0.466
    elif 4.0e3 <= Re < 4.0e4:
        C, m = 0.193, 0.618
    elif 4.0e4 <= Re < 4.0e5:
        C, m = 0.027, 0.805
    else:
        C, m = 0.027, 0.805
    return C, m

def h_externe_cylindre(Re, Pr, lamb, D):
    C, m = hilpert_C_m(Re)
    Nu = C * (Re**m) * (Pr**0.35)
    h = Nu * lamb / D
    return h

# H2 : convection interne dans une conduite (4 cas)
def h_interne_conduite(Re, Pr, lamb, Dh,
                       condition_limite="Tparoi",
                       turbulence_mode="auto"):
    if Re < 2500.0:
        if condition_limite == "flux":
            Nu = 4.363
        else:
            Nu = 3.66
    else:
        if turbulence_mode == "Dittus":
            Nu = 0.023 * (Re**0.8) * (Pr**0.4)
        elif turbulence_mode == "Colburn":
            Nu = 0.023 * (Re**0.8) * (Pr**(1.0/3.0))
        else:
            if 0.7 < Pr < 150.0:
                Nu = 0.023 * (Re**0.8) * (Pr**0.4)
            elif Pr > 0.5:
                Nu = 0.023 * (Re**0.8) * (Pr**(1.0/3.0))
            else:
                Nu = 0.023 * (Re**0.8) * (Pr**0.4)

    h = Nu * lamb / Dh
    return h

# 5) MODÈLE Thermo

N = int(L / Dx)

# Débit d’air par tranche dx (répartition uniforme sur la longueur)
qair_slice = qair * (Dx / L)

# Profil d'entrée air initial : 280 K partout (N valeurs)
T_air_in_x = np.full(N, Tair, dtype=float)

# stocker toutes les sorties par étage
T_air_out_by_stage = []

for stage in range(N_stages):

    # H2 repart à 40 K à chaque étage
    T_H2_x = np.zeros(N + 1, dtype=float)
    T_H2_x[0] = TH2

    # sortie air de cet étage (N valeurs)
    T_air_out_x = np.zeros(N, dtype=float)

    for i in range(N):
        T_air_in_i = T_air_in_x[i]
        T_H2_i = T_H2_x[i]

        # propriétés AIR
        lam_air_i = lambda_air(T_air_in_i)
        cp_air_i  = Cp_air(T_air_in_i)
        mu_air_i  = visc_air(T_air_in_i)
        rho_air_i = rho_air(T_air_in_i, Pair)
        Pr_air_i  = Pr_air(T_air_in_i)

        # propriétés H2
        lam_H2_i = lambda_H2(T_H2_i)
        cp_H2_i  = Cp_H2(T_H2_i)
        mu_H2_i  = visc_H2(T_H2_i)
        rho_H2_i = rho_H2(T_H2_i)
        Pr_H2_i  = Pr_H2(T_H2_i)

        # Reynolds
        Re_air_i, _ = reynolds(qair, rho_air_i, mu_air_i, Dtubeext, A_flow_air)
        Re_H2_i,  _ = reynolds(qH2,  rho_H2_i,  mu_H2_i, Dtubeint, A_flow_H2)

        # h
        h_air_i = h_externe_cylindre(Re_air_i, Pr_air_i, lam_air_i, Dtubeext)
        h_H2_i  = h_interne_conduite(Re_H2_i, Pr_H2_i, lam_H2_i, Dtubeint,
                                     condition_limite="Tparoi",
                                     turbulence_mode="auto")

        # résistances (conv ext + conduction cyl + conv int)
        A_air_seg = Ntube * pi * Dtubeext * Dx
        A_H2_seg  = Ntube * pi * Dtubeint * Dx

        R_conv_air = 1.0 / (h_air_i * A_air_seg)
        R_conv_H2  = 1.0 / (h_H2_i  * A_H2_seg)
        R_cond = np.log(Dtubeext / Dtubeint) / (2.0 * pi * lambda_inconel * Dx * Ntube)

        R_eq = R_conv_air + R_cond + R_conv_H2

        # flux
        deltaT = T_air_in_i - T_H2_i
        Phi = deltaT / R_eq

        # air sortant de cet étage à la maille i
        T_air_out_x[i] = T_air_in_i - Phi / (qair_slice * cp_air_i)

        # H2 se réchauffe le long de x dans cet étage
        T_H2_x[i + 1] = T_H2_i + Phi / (qH2 * cp_H2_i)

    # préparation étage suivant
    T_air_out_by_stage.append(T_air_out_x.copy())
    T_air_in_x = T_air_out_x.copy()

# 6) RÉSULTATS

T_air_out_list_final = T_air_in_x.tolist()

print("\nListe des T_air_sortie après", N_stages, "étages (", len(T_air_out_list_final), "valeurs) :")
print(T_air_out_list_final)

print("\nAir sortie min/max [K] =", float(np.min(T_air_in_x)), float(np.max(T_air_in_x)))