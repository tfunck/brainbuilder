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
    int n_new_elements=100000;

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


long unsigned int** reformat_polygons(int npoly, long unsigned int *polygons_flat){
    long unsigned int** polygons=malloc(npoly*sizeof(*polygons));
    int index=0;
    for(int i = 0; i < npoly; i++) {
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

unsigned int find_opposing_vertex(float* cur_poly, float** polygons, int npoly){

}


int upsample_polygons(long unsigned int*** polygons_ptr, float*** vtx_coords,unsigned long int*** root_idx, int npoly, int* nvtx, int* n_alloc_to_poly,  int* n_alloc_to_vtx, const float resolution, Coords3D* coords3d){
    float max_dist=99999;
    int coord_offset=*nvtx;
    int faces_offset=*npoly;
    int coord_idx=0;

    while( max_dist > resolution ){
        for(int i=0; i<npoly;i++){
            if ( i % 100 == 0 ) printf("%3.1f\r",(float)i/npoly*100);
            float poly_max_dist = calc_max_dist( (*vtx_coords)[polygons[i][0]], (*vtx_coords)[polygons[i][1]], (*vtx_coords)[polygons[i][2]] );
            max_dist = max_dist > poly_max_dist ? max_dist : poly_max_dist ; 

            //subdivide(polygons[i], new_poly_idx, root_idx, nvtx, &total_n_poly, vtx_coords, n_alloc_to_vtx, resolution, coords3d);
            if ( poly_max_dist > resolution ) {
            
                int index = coord_offset + coord_idx;
                coord_idx += 1;
                
                j = opposing_polygons[i];

                if len_ar == -1 : continue

                opposing_index = get_opposing_poly_index();
                //the idea is that the new coordinate that was interpolated between vertex a and b is stored in new_coords[index]

                new_faces[counter] = sorted([a,index,c]) ;
                new_faces[opposite_poly_index] = sorted([index,c,b]) ;
                new_faces[faces_offset] = sorted([a,index,d]) ;
                new_faces[faces_offset+1] = sorted([index,b,d]) ;
                
                ngh[a][ ngh[int(a)] == int(b) ] = index ;
                ngh[b][ ngh[int(b)] == int(a) ] = index ;
                ngh[c]=np.append(ngh[c], [index]) ;
                ngh[d]=np.append(ngh[d], [index]) ;
                ngh[index]=np.array(sorted([a,b,c,d])) ;

                faces_offset += 2
            }
        }
        free_2d((void**) polygons, n_alloc_to_poly_0);
        polygons=*new_polygons_ptr;
        npoly = total_n_poly;

        printf("\n%f %d\n", max_dist, total_n_poly);
    }

    free_2d((void**) new_poly_idx, 4);
    *polygons_ptr=polygons;
    return npoly;
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
    int n_alloc_to_vtx = nvtx;
    int n_alloc_to_poly = npoly;
    //long unsigned int** final_polygons = (long unsigned int**) alloc_2d(n_alloc_to_poly, sizeof(*final_polygons),sizeof(**polygons));
    //Iterate over polygons and subsample them
    int precision=2;
    unsigned long int** root_idx=init_root_idx(nvtx);
    Coords3D coords3d=init_coords3d( vtx_coords, nvtx, precision) ;

    int new_npoly = upsample_polygons(&polygons,  &vtx_coords, &root_idx, npoly, &nvtx, &n_alloc_to_poly, &n_alloc_to_vtx, resolution, &coords3d);

    printf("\twriting %d vertices (v) and %d polygons (p) to %s\n",nvtx, new_npoly, out_fn);
    write_output(out_fn, vtx_coords, root_idx, polygons, nvtx, new_npoly  );
    
    //free_2d((void**) polygons, npoly); 
    free_2d((void**) root_idx, n_alloc_to_vtx ); 
    free_2d((void**)vtx_coords, n_alloc_to_vtx);
    free_2d((void**) polygons,n_alloc_to_poly);
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
            printf("\t%d %d\n",root_index[i][0],root_index[i][1]);
            resampled_coords[i][j] = (resampled_coords[root_index[i][0]][j] + resampled_coords[root_index[i][1]][j])/2.;
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
