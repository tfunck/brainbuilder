#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

void** alloc_2d(int n, unsigned int size_1, unsigned int size_2){
    void **array = malloc(size_1*n);
    for (int i=0; i<n; i++) array[i]=malloc(size_2*3);
    return array;
}
float mag(float* v1,float* v2){
    float out=0;
    for(int i=0; i<3; i++){
            out += (v1[i]-v2[i]) * (v1[i]-v2[i]);
    }
    return sqrt(out);

}

float calc_max_dist(float* v1, float *v2, float *v3 ) {
    float a=mag(v1,v2);
    float b=mag(v1,v3);
    float c=mag(v2,v3);
    float max=0;
    if ( a >= b && a >= c) max=a;
    else if ( b >= a && b >= c)  max = b;
    max = c;
    //printf("%fmm\n",max);
    return max;
}

int vertex_in_array(float* vtx, float** ar, int n){
    for(int i=0; i<n; i++){
        if( (vtx[0] == ar[i][0]) && (vtx[1] == ar[i][1]) && (vtx[2] == ar[i][2]) ) return 1;
    }
    return 0;
}
int add_vertex(float *vtx, int idx, int *n, float*** final, int *n_alloc_to_final){
    int n_new_elements=40000;

    if(*n >= *n_alloc_to_final){
       *final = realloc(*final, (*n_alloc_to_final + n_new_elements) * sizeof(**final));
       for(int j = *n_alloc_to_final; j < *n_alloc_to_final + n_new_elements; j++){
            (*final)[j] = malloc(sizeof(***final)*3);   
       }
       *n_alloc_to_final = *n_alloc_to_final + n_new_elements;
    }

    (*final)[idx][0] = vtx[0];
    (*final)[idx][1] = vtx[1];
    (*final)[idx][2] = vtx[2];
    return 0;
}

int add_polygon(long unsigned int *cur_poly,  int *n, long unsigned int*** final, int *n_alloc_to_final){
    int n_new_elements=40000;

    for(int i=0; i < 3; i++){
       if(*n >= *n_alloc_to_final){
           *final = realloc(*final, (*n_alloc_to_final + n_new_elements) * sizeof(**final));
           for(int j = *n_alloc_to_final; j < *n_alloc_to_final + n_new_elements; j++){
                (*final)[j] = malloc(sizeof(***final)*3);   
           }
           *n_alloc_to_final = *n_alloc_to_final + n_new_elements;
       }

       (*final)[*n][i] = cur_poly[i];

    }
    *n = *n + 1;
    return 0;
}

unsigned long int subdivide(float** cur_poly, float*** new_poly, long unsigned int* cur_poly_idx, long unsigned int*** new_poly_idx, int* n_new_poly, int* nvtx, int* total_n_poly, unsigned long int*** final_polygons, float*** vtx_coords, int* n_alloc_to_vtx, int* n_alloc_to_poly, const float resolution){
    // First step is to define 4 new points in the current triangle.
    // The first 3 are on edges between existing points of triangle.
    // 4th point is in the center of the triangle.
    /*    
    //                  (2)
    //                  c_1
    //                 //.\\
    //                // . \\
    //               //  .  \\
    //              //p1 .p2 \\
    //             //    . (6)\\
    //    (1) n_0 //....n_3....\\ n_1 (3)
    //           // p0 . . . p3 \\
    //          //  .    .   .   \\
    //         // .  p5  . p4  .  \\
    //        c_0 ================ c_2 
    //      (0)         n_2           (4)   
    //                  (5)
    //   c_i = coords[i], original coordinate points of triangle polygon
    //   n_i = new_coords[i], subsampled coordinate points of triangle
    //   p_i = new_poly_i, index number of new subsampled triangles         
    */
    float new_coords[4][3];
    long unsigned int new_idx[4];

    //define new coordinates within polygon
    for (int i=0; i < 4; i++){
        int j = (i+1)%3;
        if( i<3) {
            for(int p=0; p<3; p++){
                new_coords[i][p] = ((*vtx_coords)[cur_poly_idx[i]][p] + (*vtx_coords)[cur_poly_idx[j]][p])/2;
                //new_coords[i][p] = (cur_poly[i][p] + cur_poly[j][p])/2;
            }
        }
        else {
            for(int p=0; p<3; p++){
                new_coords[i][p]=( (*vtx_coords)[cur_poly_idx[0]][p]+(*vtx_coords)[cur_poly_idx[1]][p]+(*vtx_coords)[cur_poly_idx[2]][p])/3. ;
                //new_coords[i][p]=(cur_poly[0][p]+cur_poly[1][p]+cur_poly[2][p])/3. ;
            }
        }

        new_idx[i]=*nvtx;
        
        if( vertex_in_array(new_coords[i], *vtx_coords, *nvtx) == 0){
            *nvtx += 1;
            add_vertex(new_coords[i], new_idx[i], nvtx, vtx_coords, n_alloc_to_vtx);
        }
    }

    // Next step is to create lists with indices/coords for the points of subdivided triangles
    int temp_poly_idx[7] = {cur_poly_idx[0], new_idx[0], cur_poly_idx[1], new_idx[1], cur_poly_idx[2], new_idx[2], new_idx[3] };
    /*float* temp_coords[7]; 
    temp_coords[0]=vtx_coords[cur_poly_idx[0]];
    temp_coords[1]=new_coords[0];
    temp_coords[2]=vtx_coords[cur_poly_idx[1]];
    temp_coords[3]=new_coords[1];
    temp_coords[4]=vtx_coords[cur_poly_idx[2]];
    temp_coords[5]=new_coords[2];
    temp_coords[6]=new_coords[3];
    */
    for(int i=0; i< 6; i++){ 
        int j = (i+1)%6;
        //if ( calc_max_dist(temp_coords[i],temp_coords[j],temp_coords[6] ) > resolution ) {
            //reallocate memory for polygon coordinates
        //new_poly = realloc(new_poly, sizeof(*new_poly) * (int) (*n_new_poly + 1));
        //new_poly[*n_new_poly ] = (float**) alloc_2d(3,sizeof(**new_poly),sizeof(***new_poly));
        
        //reallocate memory for polygon indices
        *new_poly_idx = realloc(*new_poly_idx, sizeof(**new_poly_idx) * (int) (*n_new_poly + 1) );
        (*new_poly_idx)[*n_new_poly ] = malloc(3*sizeof(***new_poly_idx));

        //FIXME: DONT ACTUALLY NEED new_poly anymore, just use poly_idx and vtx_coords
        //for(int q=0; q<3; q++){
        //    new_poly[*n_new_poly ][0][q]=temp_coords[i][q];
        //    new_poly[*n_new_poly ][1][q]=temp_coords[j][q];
        //    new_poly[*n_new_poly ][2][q]=temp_coords[6][q];
        //}

        (*new_poly_idx)[ *n_new_poly ][0] = temp_poly_idx[i];
        (*new_poly_idx)[ *n_new_poly ][1] = temp_poly_idx[j];
        (*new_poly_idx)[ *n_new_poly ][2] = temp_poly_idx[6];
        *n_new_poly += 1;
        //} else{
        //    long unsigned int poly_to_add[3]={temp_poly_idx[i], temp_poly_idx[j], temp_poly_idx[6]};
        //    add_polygon(poly_to_add,total_n_poly, final_polygons, n_alloc_to_poly);
        //}
    }

    return new_poly_idx ;
}

long unsigned int** reformat_polygons(int nvtx, long unsigned int *polygons_flat){
    long unsigned int** polygons=malloc(nvtx*sizeof(*polygons));
    int index=0;
    for(int i = 0; i < nvtx; i++) {
        polygons[i] = malloc(3*sizeof(**polygons));
        for( int j = 0; j<3; j++){
            polygons[i][j] = polygons_flat[index] ;    
            index += 1; 
        }
    }
    return polygons;
}

float** reformat_coords(int nvtx, float* vtx_coords_flat){

    float** vtx_coords=malloc(sizeof(*vtx_coords)*nvtx);
    
    if ( vtx_coords == NULL ) {printf("Memory not allocated"); exit(1);}
    
    for(int i=0; i<nvtx ; i++){
        vtx_coords[i]=malloc(sizeof(**vtx_coords)*3);
        if ( vtx_coords[i] == NULL ) {printf("Memory not allocated"); exit(1);}
        vtx_coords[i][0] =vtx_coords_flat[i*3];
        vtx_coords[i][1] =vtx_coords_flat[i*3+1];
        vtx_coords[i][2] =vtx_coords_flat[i*3+2];
    }
    return vtx_coords;
}



void free_2d(void** ar, int n){
    for(int i=0; i < n; i++){
        free(ar[i]);
    }
    free(ar);
}



int upsample_polygons(long unsigned int** polygons, long unsigned int ***final_polygons, float*** vtx_coords, int npoly, int* nvtx, int* n_alloc_to_poly,  int* n_alloc_to_vtx, const float resolution){
    int total_n_poly=0;
    int n_new_poly=0;
    printf("\n");
    for(int i=0; i<npoly;i++){
        printf("\r%f", (float) 100.* i/npoly);
        int n_cur_poly=1;
        long unsigned int** cur_poly_idx = (long unsigned int**) alloc_2d(1, sizeof(*cur_poly_idx), sizeof(**cur_poly_idx));
        float*** cur_poly =NULL;
        //float*** cur_poly =malloc(sizeof(*cur_poly));
        //cur_poly[0] = (float**) alloc_2d(3, sizeof(**cur_poly), sizeof(***cur_poly));
        float*** new_poly=NULL;
        long unsigned int** new_poly_idx=NULL;

        //Assign initial coordinate values
        for( int j=0; j < 3; j++) {
            /*for( int k=0; k<3; k++ ){
                cur_poly[0][j][k] = (*vtx_coords)[ polygons[i][j] ][ k ];
            }*/
            cur_poly_idx[0][j] = polygons[i][j];
        }
        //printf("%d",i);
        //Subdivide the polygon until all the longest distance between vertices is below resolution
        while(n_cur_poly > 0){
            n_new_poly=0;
            new_poly = NULL; 
            new_poly_idx = NULL; 
/*            printf("\tcur %d\n", n_cur_poly);*/
            for(int j=0; j < n_cur_poly; j++){
                if( calc_max_dist( (*vtx_coords)[cur_poly_idx[j][0]], (*vtx_coords)[cur_poly_idx[j][1]], (*vtx_coords)[cur_poly_idx[j][2]] ) > resolution ){
                    new_poly_idx = subdivide(cur_poly[j], new_poly, cur_poly_idx[j], &new_poly_idx, &n_new_poly, nvtx, &total_n_poly, final_polygons, vtx_coords, n_alloc_to_vtx, n_alloc_to_poly,  resolution);
/*                    printf("\t\tnew %d\n",n_new_poly);*/
                } else {
                    //add existing polygon that is below resolution
                    add_polygon(cur_poly_idx[j], &total_n_poly, final_polygons, n_alloc_to_poly);
                }

            }
            //for(int j=0; j < n_cur_poly; j++) free_2d((void**) cur_poly[j],3); 
            //free(cur_poly);
            free_2d((void**)cur_poly_idx, n_cur_poly);
            //cur_poly = new_poly;
            cur_poly_idx = new_poly_idx;
            n_cur_poly = n_new_poly;
        }
        //printf("\n");
        //for(int j=0; j < n_new_poly; j++) free_2d( (void**) new_poly[j],3); 
        //free(new_poly);
        free_2d((void**)cur_poly_idx, n_new_poly);
        
    }
    printf("\n");
    return total_n_poly;
}

int upsample(int nvtx, int npoly, float* vtx_coords_flat, long unsigned int* polygons_flat, const float resolution, char* out_fn){
    float** vtx_coords = reformat_coords(nvtx, vtx_coords_flat);
    long unsigned int** polygons = reformat_polygons(npoly, polygons_flat );
    int n_alloc_to_vtx=nvtx;
    int n_alloc_to_poly=npoly;
    long unsigned int** final_polygons = (long unsigned int**) alloc_2d(n_alloc_to_poly, sizeof(*final_polygons),sizeof(**polygons));
    //Iterate over polygons and subsample them
    int new_npoly = upsample_polygons(polygons, &final_polygons, &vtx_coords, npoly, &nvtx, &n_alloc_to_poly, &n_alloc_to_vtx, resolution);

    printf("\twriting %d vertices (v) and %d polygons (p) to %s\n",nvtx, new_npoly, out_fn);
    FILE* out_file = fopen(out_fn, "w");
    for(int i=0; i<nvtx;i++){
        fprintf(out_file,"v,%f,%f,%f\n",vtx_coords[i][0],vtx_coords[i][1],vtx_coords[i][2]);
    } 

    for(int i=0; i<new_npoly;i++){
        fprintf(out_file,"p,%ld,%ld,%ld\n",final_polygons[i][0],final_polygons[i][1],final_polygons[i][2]);
    } 

    fclose(out_file); 
    
    for(int i=0; i<npoly;i++){
        free(polygons[i]);
    }
    free(polygons);
    

    free_2d((void**)vtx_coords, n_alloc_to_vtx);

    free_2d((void**)final_polygons,n_alloc_to_poly);

   
    printf("\r\tdone.\n"); 
    fflush(stdout);
    return 0;
}

int main(int argc, char** argv){
    float vtx_coords_flat[]={0,0,0, 10,0,0, 0,10,0, 10,10,0 };
    long unsigned int polygons[]={0,1,2, 1,2,3};
    const float resolution=1;
    char* out_fn="test_upsample.txt";
    int nvtx=4;
    int npoly=2;
    printf("Test 1\n");
    upsample(nvtx, npoly, vtx_coords_flat, polygons, 1, out_fn);
    printf("Test 10\n");
    upsample(nvtx, npoly, vtx_coords_flat, polygons, 20, out_fn);
    return 0;
}
