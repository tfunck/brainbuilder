import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
from mesh_io import load_mesh_geometry, save_mesh_geometry, save_mesh_data, save_obj, read_obj

#            o (2,1)_4
#           /|
#          / |
#         /  |  
#(1,0)_2 o___o (1,1)_3
#        |  /|
#        | / | 
#        |/  |
#(0,0)_0 o___o (0,1)_1
#      

test_coords=[[0,0],    #0
            [0,1],     #1
            [1,0],     #2
            [1,1],     #3
            [2,1]]     #4
test_ngh=[  [1,2,3],   #0
            [0,3],     #1
            [0,3],     #2
            [0,1,2,4], #3
            [2,3]      #4
              ]


def get_poly_idx(idx ):
    idx.sort()
    idx = [str(i) for i in idx]
    return ''.join(idx)

class Surface():
    def __init__(self, fn=None):
        self.fn=fn
        if fn==None :
            self.ngh_list = test_ngh
            self.coords_list = test_coords
        else :
            self.mesh_dict = load_mesh_geometry(fn)
            self.coords_list = self.mesh_dict['coords']
            self.ngh_list = self.mesh_dict['neighbours']
        self.vertices={}
        self.polygons={}
        self.max_index = len( self.coords_list )
        
        self.gen_vertices()

        self.gen_polygons()
        self.n_poly=len(self.polygons.values())

        print('File has {} vertices and {} polygons'.format( len(self.vertices.values()), len(self.polygons.values())))
        self.current_max_edge_length = self.get_max_edge_length() 
        
    def gen_vertices(self):
        for coords_idx, (coords, ngh) in enumerate(zip(self.coords_list, self.ngh_list)) :
            self.vertices[coords_idx] = Vertex(coords, ngh, coords_idx)


    def gen_polygons(self):
        for idx, vtx1 in  self.vertices.items() :
            for idx2 in vtx1.ngh :
                vtx2 = self.vertices[idx2]
                polygon_coords = list( set(vtx1.ngh) & set(vtx2.ngh) )
                for vtx3_idx in polygon_coords :
                    vtx3 = self.vertices[ vtx3_idx ]
                    self.polygons[get_poly_idx([vtx1.index,vtx2.index,vtx3.index])] = Polygon([vtx1, vtx2, vtx3])
    
    def save_2d_image(self, out_fn):

        plt.subplot(1,2,1)
        for vtx in self.vertices.values() :
            plt.scatter(vtx.coords[0],vtx.coords[1],c='r')
            for ngh in vtx.ngh : 
                tgt=self.vertices[ngh]
                plt.plot([vtx.coords[0],  tgt.coords[0] ], [vtx.coords[1],  tgt.coords[1]] , c='b')

        plt.savefig(out_fn)
        plt.clf()
        plt.cla()

    def get_max_edge_length(self):
        if len(self.polygons.values()) == 0 : 
            return self.current_max_edge_length
        else :
            return max([ poly.max_edge_length for poly in self.polygons.values()])

    def upsample(self, global_max_edge_length=-1):
        
        while self.current_max_edge_length > global_max_edge_length and self.n_poly > 0 :
            new_polygons={}
            to_remove=[]

            tstart = time.time()
            for idx, (i, polygon) in enumerate(self.polygons.items()):
                if polygon.max_edge_length > global_max_edge_length or global_max_edge_length<0 :

                    if idx % 1000 == 0 : print('{:3.3%}'.format(idx/self.n_poly),end='\r')

                    self.vertices, new_polygons, self.max_index = polygon.subdivide(self.vertices,new_polygons, self.max_index, global_max_edge_length)
                    

                    to_remove.append(i)

            for i in to_remove: del self.polygons[i]

            self.polygons.update(new_polygons)

            del new_polygons
            self.current_max_edge_length = self.get_max_edge_length() 

            

            self.n_poly=len(self.polygons.values())
            n_vtx=len(self.vertices.values())
            
            tend = time.time()
            print('n polygons: {}\tn vertices {}\tmax length {:3.3}\t time {}min'.format(self.n_poly, n_vtx, self.current_max_edge_length, np.round((tend-tstart)/60.,2) ))

            # if we are not usign global_max_edge_lendth (i.e., it is negative) then break
            if global_max_edge_length < 0 : break

    def to_filename(self, fn=None):
        if fn==None:
            base,ext = os.path.splitext(self.fn)
            fn = '{}_{}.{}'.format(base, self.current_max_edge_length, ext)

            if os.path.exists(fn) :
                print('Output filename not provided and default output filename already exists: {}', fn)
                exit(1)
        print('Writing to', fn)

        coords = np.array([ vtx.coords for vtx in self.vertices.values() ] )
        ngh =  [ vtx.ngh for vtx in self.vertices.values() ] 
        ngh_count = np.array( [ len(n) for n in ngh ])
        faces = np.array([ poly.index for poly in self.polygons.values() ])
        save_mesh_geometry(fn, {'coords':coords, 'neighbours':ngh, 'neighbour_counts':ngh_count, 'faces':faces})
        
        return 0

class Polygon():
    def __init__(self, vertices):
        self.vertices = vertices
        self.index = [vtx.index for vtx in self.vertices]
        self.max_edge_length = self.set_max_edge_length()

    def set_max_edge_length(self):
        max_edge_length=0
        for i in range(3):
            j=(i+1)%3
            d = np.sum(np.abs(np.array(self.vertices[i].coords) - np.array(self.vertices[j].coords)))
            if d > max_edge_length : 
                max_edge_length = d

        return max_edge_length

    def subdivide(self,all_vertices,all_polygons,max_index, global_max_edge_length):
        # First step is to define 4 new points in the current triangle.
        # The first 3 are on edges between existing points of triangle.
        # 4th point is in the center of the triangle.
        new_coords=[]
        new_index=[]
        for i in range(3) :
            j=(i+1)%3
            new_coords.append( (np.array(self.vertices[i].coords) + np.array(self.vertices[j].coords))/2  )
            max_index += 1
            new_index.append(max_index)

        max_index += 1
        new_index.append(max_index)
        new_coords.append( (np.array(self.vertices[0].coords)+np.array(self.vertices[1].coords)+np.array(self.vertices[2].coords))/3. )

        qc=False
        if qc :
            x0 = [ self.vertices[0].coords[0], self.vertices[1].coords[0],self.vertices[2].coords[0]]
            y0 = [ self.vertices[0].coords[1], self.vertices[1].coords[1],self.vertices[2].coords[1]]
            x1 = [ new_coords[0][0], new_coords[1][0], new_coords[2][0], new_coords[3][0] ]
            y1 = [ new_coords[0][1], new_coords[1][1], new_coords[2][1], new_coords[3][1] ]
            plt.scatter( x0, y0 , c='r')
            plt.scatter( x1, y1 , c='b')
            plt.show()

        # Next step is to create lists with indices/coords for the points of subdivided triangles
        curr_poly_index = [self.vertices[0].index, new_index[0], self.vertices[1].index, new_index[1], self.vertices[2].index, new_index[2], new_index[3] ]
        curr_poly_coords = [self.vertices[0].coords, new_coords[0], self.vertices[1].coords, new_coords[1], self.vertices[2].coords, new_coords[2], new_coords[3] ]
        curr_poly_ngh = [   [1,5,6], #0
                            [0,2,6], #1
                            [1,3,6], #2
                            [2,4,6], #3
                            [3,5,6], #4
                            [0,4,6], #5
                            [0,1,2,3,4,5]]
        
        #                  (2)
        #                  c_1
        #                 //.\\
        #                // . \\
        #               //  .  \\
        #              //p1 .p2 \\
        #             //    . (6)\\
        #    (1) n_0 //....n_3....\\ n_1 (3)
        #           // p0 . . . p3 \\
        #          //  .    .   .   \\
        #         // .  p5  . p4  .  \\
        #        c_0 ================ c_2 
        #      (0)         n_2           (4)   
        #                  (5)
        #   c_i = coords[i], original coordinate points of triangle polygon
        #   n_i = new_coords[i], subsampled coordinate points of triangle
        #   p_i = new_poly_i, index number of new subsampled triangles         

        base_set = {0, 1, 2} 
        for i in range(3) :
            j, k = base_set - {i}
            #Updating existing vertices in triangle and remove their old neighbours
            try :
                self.vertices[i].ngh.remove(self.vertices[j].index)
            except ValueError :
                pass

            try :
                self.vertices[i].ngh.remove(self.vertices[k].index)
            except ValueError :
                pass

        # Using the information on the new points, their indices, and neighours, we can 
        # create new Vertex instances for these new points
        new_vertices=[]
        for counter, (coords, index, ngh_idx) in enumerate(zip(curr_poly_coords, curr_poly_index, curr_poly_ngh) ):
            ngh = [curr_poly_index[i] for i in ngh_idx]

            vtx=Vertex(coords, ngh, index)
            new_vertices.append(vtx)
            all_vertices = vtx.add_vertex_to_mesh(all_vertices )
        
        
        for i in range(6): 
            j=(i+1)%6
            vtx1 = new_vertices[i]
            vtx2 = new_vertices[(i+1)%6]
            vtx3 = new_vertices[6]
            poly = Polygon([vtx1, vtx2, vtx3])
            if poly.max_edge_length > global_max_edge_length :
                all_polygons[get_poly_idx([vtx1.index,vtx2.index,vtx3.index])] = poly


        return all_vertices, all_polygons, max_index
       
class Vertex():
    def __init__(self, coords, ngh, index):
        self.coords = coords
        self.ngh = ngh
        self.index = index

    def add_vertex_to_mesh(self, vertices) :
        # if vertex already exists in mesh, just add new neighbours to it
        try :
            vertices[ self.index ].ngh += self.ngh
            return vertices
        except KeyError :
            pass 
        
        vertices[ self.index ] = self 

        return vertices


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest='input_fn', type=str, help='Filename for input obj file.')
    parser.add_argument('-o', dest='output_fn', type=str, default=None,  help='Filename for output obj file.')
    parser.add_argument('-m', dest='max_length', default=-1, type=float, help='Maximum vertex edge length (Default of -1 -> subdivide each polygon once).')
    parser.add_argument('-t', dest='test', default=False, action='store_true', help='Run simple toy example to test code.')

    args = parser.parse_args()

    if args.test:
        surf = Surface()
        surf.save_2d_image('test_0.png')
        surf.upsample()
        surf.save_2d_image('test_1.png')
        exit(0)

    surf = Surface(args.input_fn)
    surf.upsample(args.max_length)
    surf.to_filename(args.output_fn)

