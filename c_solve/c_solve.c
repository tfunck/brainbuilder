#include <stdio.h>
#include <math.h>
#include <gsl/gsl_poly.h>
#include <Python.h>
#include <numpy/arrayobject.h>



static PyObject*  c_solve(PyObject* self,  PyObject* args) {
	PyObject* params_npy, *n_py;
	PyArg_ParseTuple(args, "Oi", &params_npy,  &n_py);  
	PyObject *params_array = PyArray_FROM_OTF(params_npy, NPY_DOUBLE, NPY_IN_ARRAY);
	int n = (int) n_py;
    int n_vtx=n/4;
    int k=0;

    double *params = (double*) PyArray_DATA(params_array);
    PyObject *pyResult = PyList_New((int)n_vtx);
    
    for(int i=0; i< n; i+=4){
        double x0=0,x1=0,x2=0,sol=0.;
        gsl_poly_solve_cubic(params[i+2]/params[i+3], params[i+1]/params[i+3], params[i+0]/params[i+3], &x0, &x1, &x2);
    
        if ( x0 > 0 && x0 <= 1) sol= x0 ;
        else if ( x1 > 0 && x1 <= 1) sol= x1 ;
        else if ( x2 > 0 && x2 <= 1) sol= x2 ;
        else sol = 0./0.;
        //printf("params %f %f %f %f\n",params[i],params[i+1],params[i+2],params[i+3]);
        //printf("%f %f %f --> %f\n",x0,x1,x2,sol);
        PyList_SetItem(pyResult, k, PyFloat_FromDouble(sol));
        k++;
    }
    PyObject* out = Py_BuildValue("O", pyResult );
	Py_DECREF(params_array);
	Py_DECREF(pyResult);
    return out;
}

static char c_solve_docs[] = "solve( ): solve cubic poly\n";
//static char surf_dist_docs[] = "surf_dist( ): Calculate minimum geodesic distance across surface\n";

static PyMethodDef c_solve_funcs[] = {
  {"solve", (PyCFunction) c_solve, METH_VARARGS, c_solve_docs},
   {NULL}
};


//#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
       PyModuleDef_HEAD_INIT,
        "c_solve",     // m_name 
        "This is the solve module",  // m_doc
        -1,                  // m_size 
        c_solve_funcs,       // m_methods
        NULL,                // m_reload 
        NULL,                // m_traverse 
        NULL,                // m_clear 
        NULL,                // m_free 
    };

PyMODINIT_FUNC
PyInit_c_solve(void){
	PyObject* m;
	m = PyModule_Create(&moduledef);
	import_array();
	return m;
}
