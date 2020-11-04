#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

struct temp_polygon;

typedef struct temp_polygon {
    float coords[3][3];
    char* index;
} polygon;


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
    return max;
}

int* vertex_in_array(float** vtx, float** ar, int n){
    int* exists=malloc(3*sizeof(int));

    for(int i=0; i < n; i++){
        for(int j=0; j<3; j++){
            if( vtx[j][0] == ar[i][0] && vtx[j][1] == ar[i][1] && vtx[j][2] == ar[i][2]){
                exists[j]=1; 
            }
        }
        if( (exists[0]==exists[1]) && (exists[0]==exists[2]) && (exists[0]==1)) return exists;
    }
    return exists;
}

float** add_vertex_to_final(float **vtx, float** ar, int *n, int *n_alloc_to_final){
    int n_new_elements=40000;

    for(int i=0; i < 3; i++){
       *n = *n + 1;
       if(*n >= *n_alloc_to_final){
           ar=realloc(ar, (*n_alloc_to_final + n_new_elements) * sizeof(*ar));
           for(int j = *n_alloc_to_final; j < *n_alloc_to_final+n_new_elements; j++){
                ar[j] = malloc(sizeof(**ar)*3);   
           }
           *n_alloc_to_final = *n_alloc_to_final + n_new_elements;
        }
        ar[*n][0] = vtx[i][0];
        ar[*n][1] = vtx[i][1];
        ar[*n][2] = vtx[i][2];
    }
    return ar;
}

polygon* subdivide(polygon cur_poly, polygon* new_poly, int* n_new_poly, float resolution, float*** final_vtx, int* n_final_vtx, int* n_alloc_to_final){
    // First step is to define 4 new points in the current triangle.
    // The first 3 are on edges between existing points of triangle.
    // 4th point is in the center of the triangle.
    float new_coords[4][3];

    //define new coordinates within polygon
    for (int i=0; i < 3; i++){
        int j=(i+1)%3;
        for(int p=0; p<3; p++){
            new_coords[i][p] = (cur_poly.coords[i][p] + cur_poly.coords[j][p])/2;
        }
    }

    for(int p=0; p<3; p++){
        new_coords[3][p]=(cur_poly.coords[0][p]+cur_poly.coords[1][p]+cur_poly.coords[2][p])/3. ;
    }

    // Next step is to create lists with indices/coords for the points of subdivided triangles
    //float* curr_poly_index = {index[0], new_index[0], index[1], new_index[1], index[2], new_index[2], new_index[3] }
    float* temp_coords[7]; // = malloc(sizeof(*temp_coords)*7); 
    temp_coords[0]=cur_poly.coords[0];
    temp_coords[1]=new_coords[0];
    temp_coords[2]=cur_poly.coords[1];
    temp_coords[3]=new_coords[1];
    temp_coords[4]=cur_poly.coords[2];
    temp_coords[5]= new_coords[2];
    temp_coords[6]=new_coords[3] ;
    //for(int i=0; i<7; i++) 
    //    printf("\ttemp_coords %f %f %f\n", temp_coords[i][0], temp_coords[i][1], temp_coords[i][2] ); 
    for(int i=0; i< 6; i++){ 
        int j = (i+1)%6;
        if ( calc_max_dist(temp_coords[i],temp_coords[j],temp_coords[6] ) > resolution ) {
            for(int q=0; q<3; q++){
                //printf("i,k = %d,%d\n",i,j);
                new_poly[*n_new_poly].coords[0][q]=temp_coords[i][q];
                new_poly[*n_new_poly].coords[1][q]=temp_coords[j][q];
                new_poly[*n_new_poly].coords[2][q]=temp_coords[6][q];
            }
            //for(int q=0; q<3; q++){
                //printf("new poly %d: %f %f %f\n", *n_new_poly, new_poly[*n_new_poly].coords[q][0], new_poly[*n_new_poly].coords[q][1], new_poly[*n_new_poly].coords[q][2] );
            //}
            *n_new_poly += 1;
            new_poly = realloc(new_poly, sizeof(*new_poly) * (int) (*n_new_poly+1));
        }
        else {
            //since the edges in the polygon are below the resolution, we can save the vertex locations
            //to an output array
            float* vtx_to_add[3];
            vtx_to_add[0]=temp_coords[i];
            vtx_to_add[1]=temp_coords[j];
            vtx_to_add[2]=temp_coords[6];
            *final_vtx = add_vertex_to_final(vtx_to_add, *final_vtx, n_final_vtx, n_alloc_to_final);
        }
    }
    
    return new_poly ;
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
}

int match_index(char* index, polygon* polygons, int n ){
    for( int i = 0 ; i<n; i++){
        if( strcmp(index,polygons[i].index)==1 ){
            return 1;
        }
    }
    return 0;
}

char* create_index(int i0, int i1, int i2){
    int max, mid, min;
    
    if (i0 > i1 && i0 > i2) max=i0;
    else if (i1 > i2 && i1 > i0) max=i1;
    else max=i2;

    if (i0 < i1 && i0 < i2) min=i0;
    else if (i1 < i2 && i1 < i0) min=i1;
    else  min=i2;

    if ( !( i0==min || i0 == max)) mid=i0;
    else if ( !( i1==min || i1 == max)) mid=i1;
    else mid=i2;


    int str_size = (int)(((ceil(log10(i0+1))+1)+(ceil(log10(i1+1))+1)+(ceil(log10(i2+1))+1)) *sizeof(char)); 
    char *index=malloc(str_size);
    sprintf(index,"%d%d%d",min,mid,max);
    return index;
}

int get_polygons(int nvtx, float** vtx_coords, long unsigned int** vtx_ngh, long unsigned int *vtx_nngh, polygon* polygons ){
    int npoly=0;
    //for vertices...
    printf("\tgetting polygons... ");
    for(int idx0=0; idx0 < nvtx; idx0++) {
        //for ngh over current vertex
        
        for(int j=0; j< vtx_nngh[idx0]; j++){
            int idx1 = vtx_ngh[idx0][j];
            //for neighours of idx1

            for(int k=0; k < vtx_nngh[idx1]; k++){
                int idx2 = vtx_ngh[idx1][k];

                if(idx0 >= nvtx || idx1 >= nvtx || idx2 >= nvtx ){
                    printf("\nError: invalid index %d %d %d, is greater than %d\n",idx0, idx1, idx2, nvtx);
                }
                
                for( int p=0; p< vtx_nngh[idx0]; p++) {
                    if (idx2 == vtx_ngh[idx0][p]){
                        //found polygon between points i, ngh_idx1, ngh_idx0
                        char* index = create_index(idx0, idx1, idx2);
                        //check if it is already in list of polygons
                        if ( match_index( index, polygons, npoly) == 0){
                            polygons[npoly].index=index;
                            for(int q=0; q<3; q++){
                                polygons[npoly].coords[0][q]=vtx_coords[idx0][q];
                                polygons[npoly].coords[1][q]=vtx_coords[idx1][q];
                                polygons[npoly].coords[2][q]=vtx_coords[idx2][q];
                                //printf("%f %f %f\n",polygons[npoly].coords[0][q], polygons[npoly].coords[1][q],polygons[npoly].coords[2][q] );
                            }
                            npoly += 1;

                        }
                    }
                }
            }
        }
    }
    printf("n polygons: %d\n", npoly);
    polygons = realloc(polygons, npoly*sizeof(*polygons) );
    return npoly;
}

long unsigned int** reformat_ngh(int nvtx, long unsigned int *vtx_ngh_flat, long unsigned int* vtx_nngh ){
    long unsigned int** vtx_ngh=malloc(nvtx*sizeof(*vtx_ngh));
    int index=0;
    for(int i = 0; i < nvtx; i++) {
        vtx_ngh[i] = malloc(vtx_nngh[i]*sizeof(**vtx_ngh));
        for( int j = 0; j<vtx_nngh[i]; j++){
            index += 1; 
            vtx_ngh[i][j] = vtx_ngh_flat[index] ;    
        }
    }
    return vtx_ngh;
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

int upsample(int nvtx, float* vtx_coords_flat, long unsigned int* vtx_ngh_flat, long unsigned int *vtx_nngh, float resolution, char* out_fn){
    polygon* polygons=malloc(sizeof(*polygons)*nvtx); 
    float** vtx_coords = reformat_coords(nvtx, vtx_coords_flat);
    long unsigned int** vtx_ngh = reformat_ngh(nvtx, vtx_ngh_flat, vtx_nngh );
    int n_poly = get_polygons( nvtx, vtx_coords, vtx_ngh, vtx_nngh, polygons);
    int n_alloc_to_final=nvtx;
    int n_final_vtx=0; 
    float** final_vtx=malloc(sizeof(*final_vtx)*n_alloc_to_final);
    float** new_vtx=malloc(sizeof(*new_vtx));
    
    for(int i=0; i<n_alloc_to_final; i++) final_vtx[i]=malloc(sizeof(**final_vtx)*3);

    printf("\tupsampling polygons\n");
    for(int i=0; i<n_poly;i++){
        //printf("\r%3.2f",(float) 100.0 *i/n_poly);
        int n_cur_poly=1;
        polygon* cur_poly = malloc(sizeof(*cur_poly)*n_cur_poly);
        polygon* new_poly=NULL;

        //Assign initial coordinate values
        for( int j=0; j < 3; j++) {
            for( int k=0; k<3; k++ ){
                cur_poly[0].coords[j][k] = polygons[i].coords[j][k];
                if(polygons[i].coords[j][k] == 0) { printf("Found 0 coordinate."); exit(0);} 
            }
        }

        //Subdivide the polygon until all the longest distance between vertices is below resolution
        while(n_cur_poly > 0){
            int n_new_poly=0;
            new_poly = malloc(sizeof(*cur_poly));
            
            for(int j=0; j < n_cur_poly; j++){
                new_poly = subdivide(cur_poly[j], new_poly, &n_new_poly, resolution, &final_vtx, &n_final_vtx, &n_alloc_to_final );
            }
            free(cur_poly); 
            cur_poly = new_poly;
            n_cur_poly = n_new_poly;
            //printf("n_cur_poly = %d\n", n_cur_poly); 
            //if (n_new_poly > 100) exit(0);
        }

        //should save the new polygon locations here
        free(new_poly);
    }
    printf("\twriting %d vertices to %s\n",n_final_vtx, out_fn);
    FILE* out_file = fopen(out_fn, "w");
    for(int i=0; i<n_final_vtx;i++){
        fprintf(out_file,"%f,%f,%f\n",final_vtx[i][0],final_vtx[i][1],final_vtx[i][2]);
    }
    fclose(out_file); 
    printf("\r\tdone.\n"); 
    //free(final_vtx);
    free(polygons);
}

