#!/usr/bin/env python

import numpy as np
import numpy.linalg as npl
import time

start_time = time.time()
np.set_printoptions(precision = 5,linewidth = 200)

def read(file):

	#Opens input files and rearranges the contents into a matrix suitable to work with
	fileCont = open('.\\input_files\\' + str(file) + '.txt', 'r') 
	content = fileCont.readlines()
	contOut = np.array([])

	if(file == 'enuc'):
		contOut = float(content[0])

	elif(file == 'eri'): 
		
		contOut = np.zeros(7**4)
		contOut = np.reshape(contOut, (7,7,7,7))
		temp = []

		for y in content:
			y = y.split('\t')
			y[-1] = y[-1].replace('\n','')
			#y = [int(y[x]) for x in range(len(y)-1)]
			y[0] = int(y[0])
			y[1] = int(y[1])
			y[2] = int(y[2])
			y[3] = int(y[3])
			y[4] = float(y[4])

			contOut[y[0]-1][y[1]-1][y[2]-1][y[3]-1] = y[4]
			contOut[y[1]-1][y[0]-1][y[2]-1][y[3]-1] = y[4]
			contOut[y[0]-1][y[1]-1][y[3]-1][y[2]-1] = y[4]
			contOut[y[1]-1][y[0]-1][y[3]-1][y[2]-1] = y[4]
			contOut[y[2]-1][y[3]-1][y[0]-1][y[1]-1] = y[4]
			contOut[y[3]-1][y[2]-1][y[0]-1][y[1]-1] = y[4]
			contOut[y[2]-1][y[3]-1][y[1]-1][y[0]-1] = y[4]
			contOut[y[3]-1][y[2]-1][y[1]-1][y[0]-1] = y[4]
		
		return contOut

	else:
		matrixDim = np.roots([0.5, 0.5, -len(content)])
		
		contOut = np.zeros(int(matrixDim[1])*int(matrixDim[1]))
		contOut = np.reshape(contOut, (int(matrixDim[1]),int(matrixDim[1])))

		for i in content:
			i = i.split('\t')
			i[-1] = i[-1].replace('\n','')
			i = [float(x) for x in i]
			contOut[int(i[0])-1][int(i[1])-1] = i[2]

			if(i[0] != i[1]):
				contOut[int(i[1])-1][int(i[0])-1] = i[2]
		contOut = np.reshape(contOut, (int(matrixDim[1]),int(matrixDim[1])))

	fileCont.close()
	return contOut

def orthMat(S):

	s, U = npl.eig(S)

	s_invsqrt = np.diag(s**(-0.5))
	S_invsqrt = np.dot(U, np.dot(s_invsqrt,np.transpose(U)))

	return S_invsqrt

def orthogonalize(M,S_invsqrt):

	M_orth = np.matmul(np.transpose(S_invsqrt), np.matmul(M,S_invsqrt))

	return M_orth

def P(C):
	P_mat = np.zeros(len(C)*len(C), dtype = float)
	P_mat = np.reshape(P_mat, (len(C),len(C)))			#creates a matrix of the right dimensions filled with zeros

	for mu in range(len(C)):
		for nu in range(len(C)):
			for m in range(5):								#change to einsum
				P_mat[mu][nu] += C[mu][m]*C[nu][m]			#Sums only over the occupied orbitals

	return P_mat	

def energy(H,F,P):

	E_e = np.sum(P * (H + F))

	return E_e

def F_new(H_core,D,rep):

	Fock = H_core + 2*np.einsum('kl,ijkl->ij',D,rep) - np.einsum('kl,ikjl->ij',D,rep)

	return Fock


def enCoef(F, S_invsqrt):

	e_o,c_o = npl.eigh(F)
	C_mo = np.matmul(S_invsqrt,c_o)

	return e_o, C_mo

def MP2(rep, E_mo, C_mo):
	"""
	rep_ao = np.einsum('ip,jq,ijkl,kr,ls->pqrs', C_mo,C_mo,rep,C_mo,C_mo)
	
	#This is equivalent to the noddy algorithm - simple but about 5 times slower than the blow algorithm for water with STO-3G
	"""
	rep_ao = np.einsum('ijkl,ls->ijks', rep, C_mo)
	rep_ao = np.einsum('ijks,kr->ijrs', rep_ao, C_mo)
	rep_ao = np.einsum('ijrs,jq->iqrs', rep_ao, C_mo)
	rep_ao = np.einsum('iqrs,ip->pqrs', rep_ao, C_mo)
	

	E_mp2 = 0

	for i in range(5):
		for j in range(5):
			for a in range(5,len(E_mo)):
				for b in range(5,len(E_mo)):

					E_mp2 += rep_ao[i][a][j][b] * (2*rep_ao[i][a][j][b]-rep_ao[i][b][j][a]) / (E_mo[i] + E_mo[j] - E_mo[a] - E_mo[b])

	return E_mp2

def main():
	enuc = read("enuc")
	H_core = read("kinetic") + read("nucAtr")
	rep = read("eri")
	S_invsqrt = orthMat(read("overlap"))
	
	Fock = orthogonalize(H_core,S_invsqrt)
	E_ao, C_mo = enCoef(Fock,S_invsqrt)
	Den = P(C_mo)
	En_e = energy(H_core,H_core,Den)
	E_old = 0.0

	while abs(E_old - En_e) > 0.000000000001:

		E_old = En_e

		Fock = F_new(H_core,Den,rep)
		F_orth = orthogonalize(Fock,S_invsqrt)
		E_ao, C_mo = enCoef(F_orth,S_invsqrt)
		Den = P(C_mo)
		En_e = energy(H_core,Fock,Den)

	MP2(rep, E_ao, C_mo)

	return None

main()
print("--- %s seconds ---" % (time.time() - start_time))