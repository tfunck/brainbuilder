#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
typedef struct temp_Coords3D{
    unsigned long int*** n;
    unsigned long int**** idx;
    int precision;
    unsigned int maxima_int[3];
} Coords3D;

void** alloc_2d(int n, unsigned int size_1, unsigned int size_2){
    void **array = malloc(size_1*n);
    for (int i=0; i<n; i++) array[i]=malloc(size_2*3);
    return array;
}
float mag(float* v1,float* v2){
    float out=0;
    for(int i=0; i<3; i++){
/*            printf("\t\t\t%d %f - %f\n",i,v1[i],v2[i]);*/
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

int convert_coord(float x, float precision, int offset){return offset+ (int) ceil(fabs((x*precision))) ; }

int vertex_in_array(float* vtx, float** ar, int n, Coords3D* coords3d){
    int x=convert_coord(vtx[0],coords3d->precision,0);
    int y=convert_coord(vtx[1],coords3d->precision,0);
    int z=convert_coord(vtx[2],coords3d->precision,0);
    int n_coords = coords3d->n[x][y][z];
    for(int i=0; i<n_coords; i++){
        int index = coords3d->idx[x][y][z][i];
        if( (vtx[0] == ar[index][0]) && (vtx[1] == ar[index][1]) && (vtx[2] == ar[index][2]) ) return index;
    }
    return -1;
}
int add_vertex(float *vtx, int idx, int *n, float*** final, int *n_alloc_to_final, Coords3D* coords3d){
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
    
    int x=convert_coord(vtx[0],coords3d->precision,0);
    int y=convert_coord(vtx[1],coords3d->precision,0);
    int z=convert_coord(vtx[2],coords3d->precision,0);

    coords3d->idx[x][y][z]=realloc(coords3d->idx[x][y][z],sizeof( unsigned long int****)*(coords3d->n[x][y][z]+1));
    coords3d->idx[x][y][z][coords3d->n[x][y][z]] = idx;
    coords3d->n[x][y][z]+=1;
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

unsigned long int** subdivide(long unsigned int* cur_poly_idx, long unsigned int*** root_idx, long unsigned int** new_poly_idx, int* n_new_poly, int* nvtx, int* total_n_poly, unsigned long int*** final_polygons, float*** vtx_coords, int* n_alloc_to_vtx,  const float resolution, Coords3D* coords3d){
    /*    
    //                  (2)
    //                  c_1
    //                 //\\
    //                //  \\
    //               //    \\
    //              //      \\
    //             //--------\\
    //    (1) n_0 // \      / \\ n_1 (3)
    //           //   \    /   \\
    //          //     \  /     \\
    //        c_0 ============== c_2 
    //      (0)         n_2           (4)   
    //                  (5)
    */
    float new_coords[3][3];
    long unsigned int new_idx[3];

    //define new coordinates within polygon
    for (int i=0; i < 3; i++){
        int j = (i+1)%3;
        for(int p=0; p<3; p++){
            new_coords[i][p] = ((*vtx_coords)[cur_poly_idx[i]][p] + (*vtx_coords)[cur_poly_idx[j]][p])/2;
        }
        //add new vertex index 
        int vtx_idx = vertex_in_array(new_coords[i], *vtx_coords, *nvtx, coords3d);

        //if the vertex is not already in the list, vtx_idx will == -1
        if(vtx_idx < 0){
            int n_alloc_to_root=*n_alloc_to_vtx;
            new_idx[i]=*nvtx;
            *nvtx += 1;
            add_vertex(new_coords[i], new_idx[i], nvtx, vtx_coords, n_alloc_to_vtx, coords3d);
            //add_root_poly(root_poly[i],new_idx[i], nvtx,   );
            if( *n_alloc_to_vtx != n_alloc_to_root){
                *root_idx = realloc(*root_idx, sizeof(**root_idx) * *n_alloc_to_vtx );
                for(int j=n_alloc_to_root; j < *n_alloc_to_vtx; j++)
                    (*root_idx)[j] = malloc(sizeof(***root_idx)*3);
            }
            (*root_idx)[*nvtx - 1][0]=cur_poly_idx[i];
            (*root_idx)[*nvtx - 1][1]=cur_poly_idx[j];

        } else{
        //if vertex is already in list, then vtx_idx == index number of that vertex
            new_idx[i]=vtx_idx;
        }
    }
    
    // Next step is to create lists with indices/coords for the points of subdivided triangles
    int temp_poly_idx[6] = {cur_poly_idx[0], new_idx[0], cur_poly_idx[1], new_idx[1], cur_poly_idx[2], new_idx[2]};
   
    new_poly_idx =(unsigned long int **) realloc(new_poly_idx, (*n_new_poly+4) * sizeof(*new_poly_idx) );
    int p[4][3]={{0,1,5},{1,2,3},{3,4,5},{1,3,5}};
    
    for(int x=0; x< 4; x++){ 
        int i = p[x][0];
        int j = p[x][1];
        int k = p[x][2];
        int n = *n_new_poly + x;

        new_poly_idx[n]=malloc(3*sizeof(**new_poly_idx));
        new_poly_idx[ n ][0] = temp_poly_idx[i];
        new_poly_idx[ n ][1] = temp_poly_idx[j];
        new_poly_idx[ n ][2] = temp_poly_idx[k];
    }
    *n_new_poly += 4;

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



int upsample_polygons(long unsigned int** polygons, long unsigned int ***final_polygons, float*** vtx_coords,unsigned long int*** root_idx, int npoly, int* nvtx, int* n_alloc_to_poly,  int* n_alloc_to_vtx, const float resolution, Coords3D* coords3d){
    int total_n_poly=0;
    int n_new_poly=0;

    for(int i=0; i<npoly;i++){
        int n_cur_poly=1;
        long unsigned int** cur_poly_idx = (long unsigned int**) alloc_2d(1, sizeof(*cur_poly_idx), sizeof(**cur_poly_idx));
        long unsigned int** new_poly_idx=NULL;

        //Assign initial coordinate values
        for( int j=0; j < 3; j++) {
            cur_poly_idx[0][j] = polygons[i][j];
        }
        //printf("%d",i);
        //Subdivide the polygon until all the longest distance between vertices is below resolution
        while(n_cur_poly > 0){
            n_new_poly=0;
            new_poly_idx = NULL; 
/*            printf("\tcur %d\n", n_cur_poly);*/
            for(int j=0; j < n_cur_poly; j++){
/*                printf("cur poly idx %d %d %d %d\n",j, cur_poly_idx[j][0], cur_poly_idx[j][1],cur_poly_idx[j][2]);*/
                float poly_max_dist = calc_max_dist( (*vtx_coords)[cur_poly_idx[j][0]], (*vtx_coords)[cur_poly_idx[j][1]], (*vtx_coords)[cur_poly_idx[j][2]] );
/*                printf("\t\t%f\n", poly_max_dist);*/
                if( poly_max_dist > resolution ){
                    new_poly_idx = subdivide(cur_poly_idx[j], root_idx, new_poly_idx, &n_new_poly, nvtx, &total_n_poly, final_polygons, vtx_coords, n_alloc_to_vtx,  resolution, coords3d);
/*                    printf("%d %d %d\n", n_new_poly, *nvtx, new_poly_idx[n_new_poly-1][0]);*/
                } else {
                    //add existing polygon that is below resolution
                    add_polygon(cur_poly_idx[j], &total_n_poly, final_polygons, n_alloc_to_poly);
                }
            }
            free_2d((void**) cur_poly_idx, n_cur_poly);
            cur_poly_idx = new_poly_idx;
            n_cur_poly = n_new_poly;
        }
        //printf("\n");
        free_2d((void**)cur_poly_idx, n_new_poly);
        
    }
    return total_n_poly;
}


Coords3D init_coords3d(float** coords, int nvtx, int precision){
    float maxima[3]={0,0,0};
    int maxima_int[3]={0,0,0};
    Coords3D coords3d;

    for(int i=0; i<nvtx; i++){
        
        for (int j=0; j<3; j++) {
            maxima[j] =  maxima[j] < fabs(coords[i][j]) ? fabs(coords[i][j]) : maxima[j];
            maxima_int[j] = convert_coord(maxima[j],precision,1);
            //if(j==0)printf("%f %f\n", coords[i][j], maxima[j]);
        }
    }
    unsigned long int*** n=malloc(sizeof(*n) * (int) maxima_int[0]);
    unsigned long int**** idx=malloc(sizeof(*idx) * maxima_int[0]);
/*    printf("maxima int %d %d %d\n", maxima_int[0], maxima_int[1], maxima_int[2]);*/
    for(int i=0; i<maxima_int[0]; i++){

        idx[i]=malloc(sizeof(**idx)*(int)maxima_int[1]);
        n[i] = malloc(sizeof(**n)*(int)maxima_int[1]);

        for (int j=0; j<maxima_int[1]; j++) {

            idx[i][j]=malloc(sizeof(***idx)*(int)maxima_int[2]);
            n[i][j]=malloc(sizeof(unsigned int***) * (int) maxima_int[2]);
            for (int k=0; k<maxima_int[2]; k++){ 
                n[i][j][k]=0;
                idx[i][j][k]=NULL;
            }
        }
    }
    for(int i=0; i<nvtx; i++){
        unsigned int x = convert_coord(coords[i][0], precision,0);
        unsigned int y = convert_coord(coords[i][1], precision,0);
        unsigned int z = convert_coord(coords[i][2], precision,0);
        idx[x][y][z]=realloc(idx[x][y][z],sizeof(****idx)*(n[x][y][z]+1));
        idx[x][y][z][ n[x][y][z]]=i;
        n[x][y][z]+=1;
    }

    coords3d.precision = precision;
    coords3d.n = n;
    coords3d.idx = idx;
    for(int i=0; i<3;i++) coords3d.maxima_int[i]=maxima_int[i];
    
    return coords3d;
}

int free_coords(Coords3D* coords3d){

    for(int x =0; x<coords3d->maxima_int[0]; x++){
        for(int y =0; y<coords3d->maxima_int[1]; y++){
            for(int z =0; z<coords3d->maxima_int[2]; z++){
                free(coords3d->idx[x][y][z]);
            }

            free(coords3d->idx[x][y]);
            free(coords3d->n[x][y]);
        }
        free(coords3d->n[x]);
        free(coords3d->idx[x]);
    }
    free(coords3d->idx);
    free(coords3d->n);
    return 0;
}

unsigned long int** init_root_idx(int nvtx){
    unsigned long int** root_idx=malloc(nvtx*sizeof(*root_idx));
    if(root_idx == NULL){printf("Error allocating memory for root_idx\n"); exit(1);}
    for(int i=0; i<nvtx; i++) { 
        root_idx[i]=malloc(3*sizeof(**root_idx));
        root_idx[i][0]=root_idx[i][1]=root_idx[i][2]=-1;
    }
    return root_idx;
}


int write_output(char* out_fn, float** vtx_coords, unsigned long int** root_idx, unsigned long int** final_polygons, int nvtx, int new_npoly  ){
    FILE* out_file = fopen(out_fn, "w");
    int n_out_lines = nvtx > new_npoly ? nvtx : new_npoly;

    for(int i=0; i<n_out_lines;i++){

        if(i<nvtx) 
            fprintf(out_file,"v,%f,%f,%f\nr,%ld,%ld,%ld\n",vtx_coords[i][0],vtx_coords[i][1],vtx_coords[i][2],root_idx[i][0],root_idx[i][1]);
/*            fprintf(out_file,"v,%f,%f,%f\nr,%ld,%ld,%ld\n",vtx_coords[i][0],vtx_coords[i][1],vtx_coords[i][2],root_idx[i][0],root_idx[i][1],root_idx[i][2]);*/

        if(i<new_npoly) 
            fprintf(out_file,"p,%ld,%ld,%ld\n",final_polygons[i][0],final_polygons[i][1],final_polygons[i][2]);
    }    
    fclose(out_file); 
    return 0;
}

int upsample(int nvtx, int npoly, float* vtx_coords_flat, long unsigned int* polygons_flat, const float resolution, char* out_fn){
    float** vtx_coords = reformat_coords(nvtx, vtx_coords_flat); 
    long unsigned int** polygons = reformat_polygons(npoly, polygons_flat ); 
    int n_alloc_to_vtx=nvtx;
    int n_alloc_to_poly=npoly;
    long unsigned int** final_polygons = (long unsigned int**) alloc_2d(n_alloc_to_poly, sizeof(*final_polygons),sizeof(**polygons));
    //Iterate over polygons and subsample them
    int precision=1;
    unsigned long int** root_idx=init_root_idx(nvtx);
    Coords3D coords3d=init_coords3d( vtx_coords, nvtx, precision) ;

    int new_npoly = upsample_polygons(polygons, &final_polygons, &vtx_coords, &root_idx, npoly, &nvtx, &n_alloc_to_poly, &n_alloc_to_vtx, resolution, &coords3d);

    printf("\twriting %d vertices (v) and %d polygons (p) to %s\n",nvtx, new_npoly, out_fn);
    write_output(out_fn, vtx_coords, root_idx, final_polygons, nvtx, new_npoly  );
    
    free_2d((void**) polygons, npoly); 
    free_2d((void**) root_idx, n_alloc_to_vtx ); 
    free_2d((void**)vtx_coords, n_alloc_to_vtx);
    free_2d((void**) final_polygons,n_alloc_to_poly);
    free_coords(&coords3d);
   
    printf("\r\tdone.\n"); 
    fflush(stdout);
    return 0;
}

int resample(int nvtx, int lores_nvtx, float* coords_flat, unsigned long int* root_index_flat,char* out_fn  ){
    float** coords = reformat_coords(lores_nvtx, coords_flat);
    float** resampled_coords = (float**) alloc_2d(nvtx, sizeof(*resampled_coords), sizeof(**resampled_coords));
    long unsigned int** root_index = reformat_polygons(nvtx, root_index_flat);

    FILE* out_file = fopen(out_fn, "w");
    for(int i=0; i<lores_nvtx; i++){
        resampled_coords[i][0] = coords[i][0];
        resampled_coords[i][1] = coords[i][1];
        resampled_coords[i][2] = coords[i][2];
        fprintf(out_file,"v,%f,%f,%f\n",resampled_coords[i][0],resampled_coords[i][1],resampled_coords[i][2]);
    }

    free_2d((void**)coords, lores_nvtx);

    //iterate over root polygons
    for( int i=lores_nvtx; i<nvtx; i++){
        //iterate over 3 dimensions 
        for( int j=0; j<3; j++){
            resampled_coords[i][j] =  (resampled_coords[root_index[i][0]][j] + resampled_coords[root_index[i][1]][j] )/2.;
        }
        fprintf(out_file,"v,%f,%f,%f\n",resampled_coords[i][0],resampled_coords[i][1],resampled_coords[i][2]);
    }
       
    fclose(out_file);
    return 0;
}

int main(int argc, char** argv){
    float vtx_coords_flat[]={0,0,0, 10,0,0, 0,10,0, 10,10,0 };
    long unsigned int polygons[]={0,1,2, 1,2,3};
    char* out_fn="test_upsample.txt";
    int nvtx=4;
    int npoly=2;
    upsample(nvtx, npoly, vtx_coords_flat, polygons, 1, out_fn);

    return 0;
}
