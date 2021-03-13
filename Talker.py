import tensorflow as tf
import numpy as np
import time

class Talker:
    def __init__(self, parameters):
        super(Talker, self).__init__()
        # Dividing vocal tract into nVocalTractAreas (both sub- and supraglottal)
        self.nVocalTractAreas = int(parameters["nVocalTractAreas"])
        # Initializing these areas for computing forward and reverse wave propagation.
        self.initializeAreas()
        # cG: Gain of elongation of vocal folds.
        self.cG = np.float32(parameters["cG"])
        # cR: Torque ratio of vocal folds.
        self.cR = np.float32(parameters["cR"])
        # cH: Adductory strain factor of vocal folds.
        self.cH = np.float32(parameters["cH"])
        # cLo: Resting vocal fold length for male.
        self.cLo = np.float32(parameters["cLo"])
        # cTo: Resting vocal fold thickness for male.
        self.cTo = np.float32(parameters["cTo"])
        # cDmo: Resting depth of muscle.
        self.cDmo = np.float32(parameters["cDmo"])
        # cDlo: Resting depth of deep layer(ligament) of vocal folds.
        self.cDlo = np.float32(parameters["cDlo"])
        # cDco: Resting depth of cover layer of vocal folds.
        self.cDco = np.float32(parameters["cDco"])
        # crho: Tissue density of airways and vocal folds.
        self.crho = np.float32(parameters["crho"])
        # cmuc: Sheer modulus of mucosa(cover) layer.
        self.cmuc = np.float32(parameters["cmuc"])
        # cmub: Sheer modulus in muscle(body) layer.
        self.cmub = np.float32(parameters["cmub"])
        # czeta: parameter for computing damping coefficients of vocal fold layers.
        self.czeta = np.float32(parameters["czeta"])
        # csnd: speed of sound.
        self.csnd = np.float32(parameters["csnd"])
        # Ce: Scaling parameter for non-linearity in fiber stress.
        self.Ce = np.float32(parameters["Ce"])
        # Fs: Sampling rate.
        self.Fs = np.float32(parameters["Fs"])
        # td: Simulation time.
        self.td = np.float32(parameters["td"])
        # ep1: Strain at which fiber linearity begins.
        self.ep1 = np.reshape(np.asarray([-0.5, -0.5, -0.5], dtype=np.float32), [1, 3])
        # ep2: Strain at which fiber non-linearity begins.
        self.ep2 = np.reshape(np.asarray([-0.35, 0.0, -0.05], dtype=np.float32), [1, 3])
        # neq: Number of forces acting on vocal force surfaces.
        self.neq = 6
        # F: Forces acting on vocal fold surfaces
        self.F = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), self.neq], dtype=np.float32), name="F")
        # f: Forward wave propagation through airways
        self.f = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), self.nVocalTractAreas], dtype=np.float32), name="f")
        # b: Backward wave propagation through airways
        self.b = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), self.nVocalTractAreas], dtype=np.float32), name="b")
        # fprev: f value at previous time sample.
        self.fprev = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="fprev")
        # bprev: b value at previous time sample.
        self.bprev = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="bprev")
        # pps: subglottal pressure.
        self.pps = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="pps")
        # ppe: supraglottal pressure.
        self.ppe = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="ppe")
        # pL: lung pressure.
        self.pL = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="pL")
        # psigmuc: fiber stress in the mucosa layer.
        self.psigmuc = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="psigmuc")
        # psigl: fiber stress in the ligament layer.
        self.psigl = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="psigl")
        # psigp: passive fiber stress in the ThyroArytenoid muscle layer.
        self.psigp = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="psigp")
        # ieplx: statring index of epilarynx for wave propagation.
        self.ieplx = int(parameters["ieplx"])
        # jmoth: ending index of mouth(vocal tract) for wave propagation.
        self.jmoth = int(parameters["jmoth"])
        # itrch: starting index for trachea for wave propagation.
        self.itrch = self.jmoth + 1
        # jtrch: ending index for trachea for wave propagation.
        self.jtrch = self.nVocalTractAreas - 1
        self.pvtatten = np.float32(parameters["pvtatten"])
        # Pox: oral pressure
        self.Pox = tf.Variable(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32), name="Pox")
        self.PI = np.float32(np.pi)
        # psigam: Maximum active stress in the ThyroArytenoid muscle layer.
        self.psigam = np.float32(parameters["psigam"])
        # pdelta: numerical fraction.
        self.pdelta = np.float32(parameters["pdelta"])
        # pAe: Epiglottal area.
        self.pAe = np.float32(parameters["pAe"])
        # Initialize airway area function (Par)
        self.initializePar()
        # rho: density of air
        self.rho = np.float32(parameters["rho"])
        self.rhoc = self.rho*self.csnd
        # Initialize fiber stress constants(sig0, sig2)
        self.initializeSigs()
        self.tcos = np.float32(parameters["tcos"])
        self.fcos = np.float32(1.0/(2.0*self.tcos))
        self.alpha = 1.0-self.pvtatten/tf.sqrt(self.par)
        # R: radiation resistance
        self.R = 128.0/(9.0*self.PI**2)
        # N: number of time samples
        self.N = int(self.Fs*self.td)
        # px01: lower glottal half-width.
        self.px01 = np.float32(parameters["x01"])
        # px02: upper glottal half-width.
        self.px02 = np.float32(parameters["x02"])
        pass
    
    def resetTalker(self, parameters):
        # Reset talker parameters for fresh estimation.
        self.F.assign(np.zeros(self.F.shape, dtype=np.float32))
        self.f.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), self.nVocalTractAreas], dtype=np.float32))
        self.b.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), self.nVocalTractAreas], dtype=np.float32))
        self.fprev.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))
        self.bprev.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))
        self.pps.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))
        self.ppe.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))
        self.pL.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))
        self.psigmuc.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))
        self.psigl.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))
        self.psigp.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))
        self.Pox.assign(np.zeros([int(parameters["nParallelBatches"]), int(parameters["nTimesteps"]), 1], dtype=np.float32))

    def initializeAreas(self):
        areas = np.asarray([0.500000000000000,0.442479345479103,0.412646562199584,0.456699123381110,0.640272511964978,0.978766755204299,1.35419211934811,1.58161061354848,1.60900889269775,1.54758331868718,1.53157366670916,
        	1.58643769136947,1.66587969233850,1.72346519200976,1.74721245682380,1.75859577143248,1.78961339878443,1.85133129120949,1.92091472586770,1.96241576346260,1.96351457363292,1.94082133384707,1.91712743402238,1.88919649802125,
            1.83618297093745,1.75745576877923,1.69602087162924,1.71085618433110,1.82655689383158,2.01244666823043,2.22572412664053,2.46441069990554,2.74357182832998,3.04448327626521,3.32307292065764,3.53354833267968,3.60907930585891,
            3.47504705278849,3.13165863634353,2.68791589968153,2.26703128207802,1.91539386902203,1.61892788597428,1.35224583849292], dtype=np.float32)
        self.areas = np.reshape(areas, [areas.shape[0], 1])
        pass

    def initializeSigs(self):
        self.sig0 = np.reshape(np.asarray([5000.0, 4000.0, 10000.0], dtype=np.float32), [1, 3])
        self.sig2 = np.reshape(np.asarray([300000.0, 13930.0, 15000.0], dtype=np.float32), [1, 3])
        pass
    
    def initializePar(self):
        self.par = np.zeros((1, 1, self.nVocalTractAreas), dtype=np.float32)
        self.par[:, :, 44:self.nVocalTractAreas] = np.asarray([2.23789298417581,2.11000028416715,2.09530849360073,2.15413183969517,2.25669248992398,2.38140094284581,2.51328339572398,2.64255608893532,2.76334662716830,2.87256227741042,2.96890524372499,3.05203491881712,
            3.12187711238898,3.17808025628450,3.21961858642321,3.24454230152343,3.24987469861483,3.23165628534016,3.18513586904632,3.10510862266483,2.98640112738150,2.82450339209519,2.61634784966640,2.36123532995466,2.06190800964535,1.72576933886600,1.36625094459160,1.00432651083939,0.670172635653033,0.404976664875760,0.340000000000000,0.313137399084765], dtype=np.float32)
        self.par[:, :, 0:44] = np.reshape(self.areas, [1, 1, 44])
        pass

    @tf.function
    def __call__(self, i, controller_outputs):
        t = i/self.Fs
        if (t <= self.tcos):
            self.pps.assign(controller_outputs[:, :, 0:1] * np.float32(40000.0) * (np.float32(1.0)+tf.cos(np.float32(2.0)*self.PI*self.fcos*t-self.PI))/np.float32(2.0))
        _a = self.rules_consconv(controller_outputs)
        _b = self.compute_translation_parameters(_a[-3], _a[-4], _a[-5])
        self.ppe.assign(self.f[:, :, self.ieplx:self.ieplx+1]+self.b[:, :, self.ieplx:self.ieplx+1])
        _c = self.calc_pressures(_b[-2], _b[1], _b[2], _b[0], _a[-3], _a[-4], _a[-5], _b[-3], _b[3])
        self.rk3m(_c[0], _a[2], _a[7], _a[0], _a[-7], _c[1], _a[3], _a[8], _a[-6], _a[-1], _a[1], _a[-2])
        _d = self.calc_flow(_b[-1])
        _e = self.compute_reflection_coefficient()
        self.compute_wave_propagation(_d, _e[0], _e[1])
        self.compute_lip_radiation()
        return tf.concat([_a[-5], tf.math.log(_a[4]), tf.math.log(_a[5]), tf.math.log(_a[6])], axis=-1)

    def calc_stress(self, peps):
        # calculate stress in the 3 vocal fold layers (mucosa, ligament and muscle).
        # sig_lin: linear stress.
        # sig_nln: non-linear stress.
        sig_lin = (-self.sig0*(peps-self.ep1))/self.ep1
        sig_nln = tf.cast(tf.math.greater(peps, self.ep2), dtype=tf.float32)*(self.sig2*(tf.exp(self.Ce*(peps-self.ep2))-1.0-self.Ce*(peps-self.ep2)))
        strs = sig_lin + sig_nln
        return strs[:, :, 0:1], strs[:, :, 1:2], strs[:, :, 2:3]

    def rules_consconv(self, controller_outputs):
        # computes dynamic strain, vocal fold length, thickness and depth, fiber stress, spring constants and mass values.
        peps = self.cG*(self.cR*controller_outputs[:, :, 1:2] - controller_outputs[:, :, 2:3]) - self.cH*controller_outputs[:, :, 3:4]
        pL = self.cLo*(1.0+peps)
        pT = self.cTo/(1.0+0.8*peps)
        pDb = (controller_outputs[:, :, 2:3]*self.cDmo + 0.5*self.cDlo)/(1.0 + 0.2*peps)
        pDc = (self.cDco + 0.5*self.cDlo)/(1.0 + 0.2*peps)
        pzn =  pT*(1 + controller_outputs[:, :, 2:3])/3.0
        psigmuc,psigl,psigp=self.calc_stress(peps)
        psigm = controller_outputs[:, :, 2:3]*self.psigam*tf.maximum(0.0,1.0-1.07*tf.square(peps - 0.4)) + psigp
        psigb = (0.5*psigl*self.cDlo + psigm*self.cDmo)/pDb 
        psigc = (psigmuc*self.cDco + 0.5*psigl*self.cDlo)/pDc
        pM = self.crho*pL*pT*pDb
        pKb = 2.0*self.cmub*pL*pT/pDb + (self.PI**2)*psigb*pDb*pT/pL
        pk1 = 2.0*self.cmuc*(pL*pT/pDc)*pzn/pT + (self.PI**2)*psigc*(pDc/pL)*pzn
        pk2 = 2.0*self.cmuc*(pL*pT/pDc)*(1.0-pzn/pT) + (self.PI**2)*psigc*(pDc/pL)*pT*(1.0-pzn/pT)
        pm1 = self.crho*pL*pT*pDc*pzn/pT
        pm2 = self.crho*pL*pT*pDc*(1.0-pzn/pT)
        a =  pzn/pT
        pkc = (0.5*self.cmuc*(pL*pDc/pT)/(1.0/3.0 - a*(1.0-a)) - 2.0*self.cmuc*(pL*pT/pDc) )*a*(1.0-a)
        pB = 2.0*0.1*tf.sqrt(pM*pKb)
        pb1 = 2.0*0.1*tf.sqrt(pm1*pk1)
        pb2 = 2.0*0.6*tf.sqrt(pm2*pk2)
        return [pkc,pB,pb1,pb2,psigmuc,psigl,psigm,pk1,pk2,pm1,pm2,pL,pT,pzn,pM,pKb]

    def compute_translation_parameters(self, pzn, pT, pL):
        # computes vocal fold osciallation parameters.
        znot = pzn/pT
        pxn = (1-znot)*(self.px01+self.F[:, :, 0:1])+znot*(self.px02+self.F[:, :, 1:2])
        ptangent = (self.px01-self.px02 + 2*(self.F[:, :, 0:1]-self.F[:, :, 1:2]))/pT
        px1 = pxn + pzn*ptangent
        px2 = pxn - (pT - pzn)*ptangent
        pzc = tf.minimum(pT, tf.maximum(0.0, pzn+pxn/(1.0e-6+ptangent))) 
        pa1 = tf.maximum(self.pdelta, 2.0*pL*px1) 
        pa2 = tf.maximum(self.pdelta, 2.0*pL*px2 ) 
        pan = tf.maximum(self.pdelta, 2.0*pL*pxn) 
        pzd = tf.minimum(pT, tf.maximum(0.0, -0.2*px1/(1e-6+ptangent))) 
        pad = tf.minimum(pa2, 1.2*pa1)
        pga = tf.maximum(0.0,tf.minimum(pa1,pa2))
        return [pzc, pa1, pa2, pan, pzd, pad, pga]

    def calc_pressures(self, pad, pa1, pa2, pzc, pzn, pT, pL, pzd, pan):
        # calculate driving pressures.
        pke = (2.0*pad/self.pAe)*(1.0- pad/self.pAe)
        ppkd = (self.pps - self.ppe)/(1.0-pke)
        pph = (self.pps + self.ppe)/2.0
        
        #--------------------------------------------------
        pa1_g_pdelta_pa2_g_pdelta = tf.cast(pa1 > self.pdelta, dtype=tf.float32) * tf.cast(pa2 > self.pdelta, dtype=tf.float32)
        pa1_g_pdelta_pa2_le_pdelta = tf.cast(pa1 > self.pdelta, dtype=tf.float32) * tf.cast(pa2 <= self.pdelta, dtype=tf.float32)
        pa1_le_pdelta_pa2_g_pdelta = tf.cast(pa1 <= self.pdelta, dtype=tf.float32) * tf.cast(pa2 > self.pdelta, dtype=tf.float32)
        pa1_le_pdelta_pa2_le_pdelta = tf.cast(pa1 <= self.pdelta, dtype=tf.float32) * tf.cast(pa2 <= self.pdelta, dtype=tf.float32)
        pa1_l_pa2 = tf.cast(pa1<pa2, dtype=tf.float32)
        pa1_ge_pa2 = tf.cast(pa1>=pa2, dtype=tf.float32)
        pzd_le_pzn = tf.cast(pzd <= pzn, dtype=tf.float32)
        pzd_g_pzn = tf.cast(pzd > pzn, dtype=tf.float32)
        pzc_ge_pzn = tf.cast(pzc >= pzn, dtype=tf.float32)
        pzc_l_pzn = tf.cast(pzc < pzn, dtype=tf.float32)
        pf1 =   pa1_g_pdelta_pa2_g_pdelta*(
                    pa1_l_pa2*(
                        pzd_le_pzn*(
                            pL*pzn*self.pps - pL*(pzn - pzd + (pad/pa1)*pzd)*ppkd
                        ) + 
                        pzd_g_pzn*(
                            pL*pzn*(self.pps - ((pad**2)/(pan*pa1))*ppkd)
                        )
                    ) +
                    pa1_ge_pa2*(
                        pL*pzn*(self.pps - ((pad**2)/(pan*pa1))*ppkd)
                    )
                ) + pa1_g_pdelta_pa2_le_pdelta*(
                    pzc_ge_pzn*(
                        pL*pzn*self.pps
                    ) + 
                    pzc_l_pzn*(
                        pL*pzc*self.pps + pL*(pzn-pzc)*pph
                    )
                ) + pa1_le_pdelta_pa2_g_pdelta*(
                    pzc_l_pzn*(
                        pL*pzc*pph + pL*(pzn-pzc)*self.ppe
                    ) + 
                    pzc_ge_pzn*(pL*pzn*pph)
                ) + pa1_le_pdelta_pa2_le_pdelta*(
                    pL*pzn*pph
                )
        pf2 =   pa1_g_pdelta_pa2_g_pdelta*(
                    pa1_l_pa2*( 
                        pzd_le_pzn*(
                            pL*(pT - pzn)*(self.pps - ppkd)
                        ) + 
                        pzd_g_pzn*(
                            pL*(pT - pzn)*self.pps - pL*( (pT-pzd) + (pad/pan)*(pzd - pzn))*ppkd)
                        ) + 
                    pa1_ge_pa2*(
                            pL*(pT-pzn)*(self.pps - (pa2/pan)*ppkd)
                        )
                ) + pa1_g_pdelta_pa2_le_pdelta*(
                    pzc_ge_pzn*(
                        pL*(pzc-pzn)*self.pps + pL*(pT-pzc)*pph
                    ) + 
                    pzc_l_pzn*(
                        pL*(pT-pzn)*pph
                    )
                ) + pa1_le_pdelta_pa2_g_pdelta*(
                    pzc_l_pzn*(
                        pL*(pT-pzn)*self.ppe
                    ) + 
                    pzc_ge_pzn*(
                        pL*(pzc-pzn)*pph + pL*(pT-pzc)*self.ppe
                    )
                ) + pa1_le_pdelta_pa2_le_pdelta*(
                    pL*(pT-pzn)*pph
                )
        return [pf1, pf2]
    
    def values_4th(self, pf1, pb1, pk1, pkc, pm1, yo):
        return (pf1-pb1*(yo[:, :, 3:4]-yo[:, :, 5:6])-pk1*(yo[:, :, 0:1]-yo[:, :, 2:3])-pkc*(yo[:, :, 0:1]-yo[:, :, 1:2]))/pm1
    
    def values_5th(self, pf2, pb2, pk2, pkc, pm2, yo):
        return (pf2 - pb2*(yo[:, :, 4:5]-yo[:, :, 5:6]) - pk2*(yo[:, :, 1:2] - yo[:, :, 2:3]) - pkc*(yo[:, :, 1:2] - yo[:, :, 0:1]))/pm2
    
    def values_6th(self, pk1, pk2, pb1, pb2, pKb, pB, pM, yo):
        return (pk1*((yo[:, :, 0:1])-(yo[:, :, 2:3])) + pk2*((yo[:, :, 1:2])-(yo[:, :, 2:3])) + pb1*(yo[:, :, 3:4]-yo[:, :, 5:6])+pb2*(yo[:, :, 4:5] - yo[:, :, 5:6]) - pKb*(yo[:, :, 2:3]) - pB*yo[:, :, 5:6])/pM

    def generate_values_list(self, pf1, pf2, pb1, pb2, pk1, pk2, pm1, pm2, pkc, pKb, pB, pM, yo):
        return tf.concat([yo[:, :, 3:4], yo[:, :, 4:5], yo[:, :, 5:6], self.values_4th(pf1, pb1, pk1, pkc, pm1, yo),
        self.values_5th(pf2, pb2, pk2, pkc, pm2, yo), 
        self.values_6th(pk1, pk2, pb1, pb2, pKb, pB, pM, yo)], axis=-1)
    
    def rk3m(self, pf1, pb1, pk1, pkc, pm1, pf2, pb2, pk2, pm2, pKb, pB, pM):
        # compute vocal fold displacements using Runge-Kutta method.
        df = self.generate_values_list(pf1, pf2, pb1, pb2, pk1, pk2, pm1, pm2, pkc, pKb, pB, pM, self.F)
        k1 = (1.0/self.Fs)*df
        df = self.generate_values_list(pf1, pf2, pb1, pb2, pk1, pk2, pm1, pm2, pkc, pKb, pB, pM, self.F + k1/2.0)
        k2 = (1.0/self.Fs)*df
        df = self.generate_values_list(pf1, pf2, pb1, pb2, pk1, pk2, pm1, pm2, pkc, pKb, pB, pM, self.F + k2/2.0)
        k3 = (1.0/self.Fs)*df
        df = self.generate_values_list(pf1, pf2, pb1, pb2, pk1, pk2, pm1, pm2, pkc, pKb, pB, pM, self.F + k3/2.0)
        k4 = (1.0/self.Fs)*df
        self.F.assign_add(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)
        pass

    def calc_flow(self, pga):
        # compute glottal flow
        Atrch = self.par[:, :, -1]
        Aphrx = self.par[:, :, 0]
        ke = 2*pga/Aphrx*(1-pga/Aphrx)
        kt = 1-ke
        Astar = (Atrch * Aphrx) / (Atrch + Aphrx)
        Q = 4.0*kt * (self.f[:, :, self.jtrch:self.jtrch+1] - self.b[:, :, self.ieplx:self.ieplx+1]) / (self.csnd*self.rhoc)
        R = (pga / Astar )
        return tf.cast(Q >= tf.constant(0.0, dtype=tf.float32), dtype=tf.float32) * ((pga*self.csnd/kt) * (-R + tf.sqrt( tf.abs(R*R + Q) ))) + tf.cast(Q < tf.constant(0.0, dtype=tf.float32), dtype=tf.float32) * ((pga*self.csnd/kt) * (R - tf.sqrt( tf.abs(R*R - Q) )))
    
    def compute_reflection_coefficient(self):
        # compute reflection coefficients
        D = self.par[:, :, 0:self.jtrch] + self.par[:, :, 1:self.jtrch+1]
        r1 = (self.par[:, :, 0:self.jtrch]-self.par[:, :, 1:self.jtrch+1])/D
        r2 = -1 * r1
        return [r1, r2]
    
    def compute_wave_propagation(self, ug, r1, r2):
        # compute wave propagation
        f = tf.transpose(self.f)
        b = tf.transpose(self.b)
        ug = tf.transpose(ug)
        pps = tf.transpose(self.pps)
        alpha = tf.transpose(self.alpha)
        par = np.transpose(self.par)
        r1 = tf.transpose(r1)
        r2 = tf.transpose(r2)
        index = tf.constant([[self.ieplx]])
        value = alpha[self.ieplx:self.ieplx+1, :, :]*b[self.ieplx:self.ieplx+1, :, :] + ug*(self.rhoc/par[self.ieplx:self.ieplx+1, :, :])
        f = tf.tensor_scatter_nd_update(f, index, value)
        index = tf.constant([[self.itrch]])
        value = 0.9*pps - 0.8*b[self.itrch:self.itrch+1, :, :]*alpha[self.itrch:self.itrch+1, :, :]
        f = tf.tensor_scatter_nd_update(f, index, value)
        f = f*alpha
        b = b*alpha
        indices = tf.constant(list(map(lambda el:[el], range(self.itrch+1, self.jtrch-1, 2))))
        Psi = tf.gather_nd(f, indices)*tf.gather_nd(r1, indices) + tf.gather_nd(b, indices+1)*tf.gather_nd(r2, indices)
        b= tf.tensor_scatter_nd_update(b, indices, tf.gather_nd(b, indices+1)+Psi)
        f=tf.tensor_scatter_nd_update(f, indices+1, tf.gather_nd(f, indices)+Psi)
        index = tf.constant([[self.jtrch]])
        value = f[self.jtrch:self.jtrch+1, :, :]*alpha[self.jtrch:self.jtrch+1, :, :]-ug*self.rhoc/par[self.jtrch:self.jtrch+1, :, :]
        b = tf.tensor_scatter_nd_update(b, index, value)
        indices = tf.constant(list(map(lambda el:[el], range(self.ieplx+1, self.jmoth-1, 2))))
        Psi = tf.gather_nd(f, indices)*tf.gather_nd(r1, indices) + tf.gather_nd(b, indices+1)*tf.gather_nd(r2, indices)
        b=tf.tensor_scatter_nd_update(b, indices, tf.gather_nd(b, indices+1)+Psi)
        f=tf.tensor_scatter_nd_update(f, indices+1, tf.gather_nd(f, indices)+Psi)
        indices = tf.constant(list(map(lambda el:[el], range(self.itrch, self.jtrch, 2))))
        Psi = tf.gather_nd(f, indices)*tf.gather_nd(r1, indices) + tf.gather_nd(b, indices+1)*tf.gather_nd(r2, indices)
        b=tf.tensor_scatter_nd_update(b, indices, tf.gather_nd(b, indices+1)+Psi)
        f=tf.tensor_scatter_nd_update(f, indices+1, tf.gather_nd(f, indices)+Psi)
        indices = tf.constant(list(map(lambda el:[el], range(self.ieplx, self.jmoth, 2))))
        Psi = tf.gather_nd(f, indices)*tf.gather_nd(r1, indices) + tf.gather_nd(b, indices+1)*tf.gather_nd(r2, indices)
        self.b.assign(tf.transpose(tf.tensor_scatter_nd_update(b, indices, tf.gather_nd(b, indices+1)+Psi)))
        self.f.assign(tf.transpose(tf.tensor_scatter_nd_update(f, indices+1, tf.gather_nd(f, indices)+Psi)))
        pass

    def compute_lip_radiation(self):
        # compute lip radiation.
        b = tf.transpose(self.b)
        am = tf.sqrt(self.par[:, :, self.jmoth:self.jmoth+1]/self.PI)
        L = (2.0*self.Fs)*8.0*am/(3.0*self.PI*self.csnd)
        a2 = - self.R - L + self.R*L
        a1 = - self.R + L - self.R*L
        b2 = self.R + L + self.R*L
        b1 = -self.R + L + self.R*L
        self.bprev.assign((1.0/b2)*(self.f[:, :, self.jmoth:self.jmoth+1]*a2+self.fprev*a1+self.bprev*b1))
        self.Pox.assign((1.0/b2)*(self.Pox*b1 + self.f[:, :, self.jmoth:self.jmoth+1]*(b2+a2) + self.fprev*(a1-b1)))
        self.fprev.assign(self.f[:, :, self.jmoth:self.jmoth+1])
        index = tf.constant([[self.jmoth]])
        value = tf.transpose(self.bprev)
        self.b.assign(tf.transpose(tf.tensor_scatter_nd_update(b, index, value)))
        pass