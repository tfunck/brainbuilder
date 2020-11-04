#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "upsample_mesh.h"

static PyObject*  c_upsample_mesh(PyObject* self,  PyObject* args) {
    printf("Entered c_upsample_mesh\n");
    char* out_fn ;
    float resolution;
    int nvtx;
	PyObject* coords_obj, *ngh_obj, *nngh_obj;

    PyArg_ParseTuple(args, "OOOsfi", &coords_obj, &ngh_obj, &nngh_obj, &out_fn, &resolution, &nvtx );
    printf("Resolution %f\n", resolution);
    printf("n vertices %d\n", nvtx);
    printf("Output file %s\n",out_fn);
	PyObject *coords_array = PyArray_FROM_OTF(coords_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject *ngh_array = PyArray_FROM_OTF(ngh_obj, NPY_INT64, NPY_IN_ARRAY);
	PyObject *nngh_array = PyArray_FROM_OTF(nngh_obj, NPY_INT64, NPY_IN_ARRAY);
    long unsigned int *ngh = (long unsigned int*) PyArray_DATA(ngh_array);
	long unsigned int *nngh = (long unsigned int*) PyArray_DATA(nngh_array);
	float *coords = (float*) PyArray_DATA(coords_array);
    upsample(nvtx, coords, ngh, nngh, resolution, out_fn) ;
	Py_RETURN_NONE;
}

static char upsample_mesh_docs[] = "upsample( ): upsample surface obj mesh\n";
//static char surf_dist_docs[] = "surf_dist( ): Calculate minimum geodesic distance across surface\n";

static PyMethodDef c_upsample_mesh_funcs[] = {
  {"upsample", (PyCFunction) c_upsample_mesh, METH_VARARGS, upsample_mesh_docs},
   {NULL}
};

static struct PyModuleDef moduledef = {
   PyModuleDef_HEAD_INIT,
    "c_upsample_mesh",     // m_name 
    "This is the c_upsample_mesh module",  // m_doc
    -1,                  // m_size 
    c_upsample_mesh_funcs,    // m_methods
    NULL,                // m_reload 
    NULL,                // m_traverse 
    NULL,                // m_clear 
    NULL,                // m_free 
};

PyMODINIT_FUNC
PyInit_c_upsample_mesh(void){
	PyObject* m;
	m = PyModule_Create(&moduledef);
	import_array();
	return m;
}
