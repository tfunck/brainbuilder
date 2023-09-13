#include <stdio.h>
#include <math.h>
#include <gsl.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "solve.c"


int solve(double** params, double** sols, int n_poly=3, int n) {
	double z[n_poly*2]
    gsl_poly_complex_workspace* ws = gsl_poly_complex_workspace_alloc(n_poly);

    for(int i=0; i< n; i++){
		gsl_poly_complex_solve(params[i], n_poly, ws, gsl_complex_packed_ptr z)         
		sols[i]=z
    }
	gsl_poly_complex_workspace_free(ws);	
    return 0
}

