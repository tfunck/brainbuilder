#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "upsample_mesh.h"

static PyObject*  c_upsample_mesh(PyObject* self,  PyObject* args) {
    char* out_fn ;
    float resolution;
    int nvtx, npoly;
	PyObject* coords_obj, *polygons_obj;

    PyArg_ParseTuple(args, "OOsfii", &coords_obj, &polygons_obj, &out_fn, &resolution, &nvtx, &npoly );
	PyObject *coords_array = PyArray_FROM_OTF(coords_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject *polygons_array = PyArray_FROM_OTF(polygons_obj, NPY_INT64, NPY_IN_ARRAY);
    long unsigned int *polygons = (long unsigned int*) PyArray_DATA(polygons_array);
	float *coords = (float*) PyArray_DATA(coords_array);
    upsample(nvtx, npoly, coords, polygons, resolution, out_fn) ;
	Py_RETURN_NONE;
}

static PyObject*  c_resample_mesh(PyObject* self,  PyObject* args) {
    char* out_fn ;
    int nvtx, lores_nvtx;
	PyObject* coords_obj, *roots_obj;

    PyArg_ParseTuple(args, "OOsii", &coords_obj, &roots_obj, &out_fn, &nvtx, &lores_nvtx );
	PyObject *coords_array = PyArray_FROM_OTF(coords_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject *root_index_array = PyArray_FROM_OTF(roots_obj, NPY_INT64, NPY_IN_ARRAY);
    long unsigned int *root_index = (long unsigned int*) PyArray_DATA(root_index_array);
	float *coords = (float*) PyArray_DATA(coords_array);

    resample(nvtx, lores_nvtx, coords, root_index, out_fn);
	Py_RETURN_NONE;
}
static char upsample_mesh_docs[] = "upsample( ): upsample surface obj mesh\n";
//static char surf_dist_docs[] = "surf_dist( ): Calculate minimum geodesic distance across surface\n";

static PyMethodDef c_upsample_mesh_funcs[] = {
  {"upsample", (PyCFunction) c_upsample_mesh, METH_VARARGS, upsample_mesh_docs},
  {"resample", (PyCFunction) c_resample_mesh, METH_VARARGS, upsample_mesh_docs},
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
