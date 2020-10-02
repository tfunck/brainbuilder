import numpy as np
import matplotlib.pyplot as plt
from mesh_io import load_mesh_geometry, save_mesh_geometry, save_mesh_data, save_obj, read_obj

def get_poly_idx(idx ):
    idx.sort()
    idx = [str(i) for i in idx]
    return ''.join(idx)

class Surface():
    def __init__(self, fn):
        self.mesh_dict = load_mesh_geometry(base_mesh_fn)
        self.coord_list = self.mesh_dict['coords']
        self.ngh_list = self.mesh_dict['neighbours']
        self.vertices={}
        self.polygons={}
        self.max_index = len( self.coord_list )
        
        self.gen_vertices()

        self.get_polygons()
    
    def gen_vertices(self):
        for coord_idx, (coords, ngh) in enumerate(zip(self.coord_list, self.ngh_list)) :
            self.vertices[coord_idx] = Vertex(coords, ngh, coord_idx)

            
    def get_polygons(self):
        for idx, vtx1 in  self.vertices.items() :
            for idx2 in vtx1.ngh :
                vtx2 = self.vertices[idx2]
                polygon_coords = list( set(vtx1.ngh) & set(vtx2.ngh) )
               
                for vtx3_idx in polygon_coords :
                    vtx3 = self.vertices[ vtx3_idx ]
 
                    self.polygons[get_poly_idx([vtx1.index,vtx2.index,vtx3.index])] = Polygon([vtx1, vtx2, vtx3])

    def upsample(self):
        temp_polygons = self.polygons.copy()
        for i, polygon in temp_polygons.items():
            self.vertices, self.polygons, self.max_index = polygon.subdivide(self.vertices, self.polygons, self.max_index)
            del self.polygons[ i ] # remove the current polygon because it 



    def to_filename(self, fn):
        
        coords = np.array([ vtx.coord for vtx in self.vertices.values() ] )
        ngh =  [ vtx.ngh for vtx in self.vertices.values() ] 
        ngh_count = np.array( [ len(n) for n in ngh ])
        faces = np.array([ poly.index for poly in self.polygons.values() ])
        save_mesh_geometry(fn, {'coords':coords, 'neighbours':ngh, 'neighbour_counts':ngh_count, 'faces':faces})
        
        return 0

class Polygon():
    def __init__(self, vertices, polygons={} ):
        self.vertices = vertices
        self.index = [vtx.index for vtx in self.vertices]
        #self.polygons = polygons

    def subdivide(self,all_vertices,all_polygons,max_index):
        # First step is to define 4 new points in the current triangle.
        # The first 3 are on edges between existing points of triangle.
        # 4th point is in the center of the triangle.
        new_coords=[]
        new_index=[]
        for i in range(3) :
            j=(i+1)%3
            new_coords.append( (self.vertices[i].coord + self.vertices[j].coord)/2  )
            max_index += 1
            new_index.append(max_index)

        max_index += 1
        new_index.append(max_index)
        new_coords.append( (self.vertices[0].coord+self.vertices[1].coord+self.vertices[2].coord)/3. )

        qc=False
        if qc :
            x0 = [ self.vertices[0].coord[0], self.vertices[1].coord[0],self.vertices[2].coord[0]]
            y0 = [ self.vertices[0].coord[1], self.vertices[1].coord[1],self.vertices[2].coord[1]]
            x1 = [ new_coords[0][0], new_coords[1][0], new_coords[2][0], new_coords[3][0] ]
            y1 = [ new_coords[0][1], new_coords[1][1], new_coords[2][1], new_coords[3][1] ]
            plt.scatter( x0, y0 , c='r')
            plt.scatter( x1, y1 , c='b')
            plt.show()

        # Next step is to create lists with indices/coords for the points of subdivided triangles
        curr_poly_index = [self.vertices[0].index, new_index[0], self.vertices[1].index, new_index[1], self.vertices[2].index, new_index[2], new_index[3] ]
        curr_poly_coords = [self.vertices[0].coord, new_coords[0], self.vertices[1].coord, new_coords[1], self.vertices[2].coord, new_coords[2], new_coords[3] ]

        idx_around_tri = [[1,5],[0,2],[1,2],[2,5],[3,5],[0,3],[0,1,2,3,4,5]]
        
        # Using the information on the new points, their indices, and neighours, we can 
        # create new Vertex instances for these new points
        new_vertices=[]
        for coords, index, ngh_idx in zip(curr_poly_coords, curr_poly_index, idx_around_tri) :
            ngh = [curr_poly_index[i] for i in ngh_idx + [6]]
            vtx=Vertex(coords, ngh, index)
            new_vertices.append(vtx)
            vertices = vtx.add_vertex_to_mesh(all_vertices )

        curr_vertices=[self.vertices[0], new_vertices[0], self.vertices[1], new_vertices[1], self.vertices[2], new_vertices[2], new_vertices[3]]

        for i,j in idx_around_tri[0:-1]: 
            vtx1 = curr_vertices[i]
            vtx2 = curr_vertices[j]
            vtx3 = curr_vertices[6]
            all_polygons[get_poly_idx([vtx1.index,vtx2.index,vtx3.index])] = Polygon([vtx1, vtx2, vtx3])
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
        #   c_i = coords[i]
        #   n_i = new_coords[i]
        #   p_i = new_poly_i         

        base_set = {0,1,2} 
        for i in range(3) :
            j, k = base_set - {i}
            #Updating existing vertices in triangle and remove their old neighbours
            try :
                self.vertices[i].ngh.remove([self.vertices[j].index])
            except ValueError :
                pass

            try :
                self.vertices[i].ngh.remove([self.vertices[k].index])
            except ValueError :
                pass
        
        return all_vertices, all_polygons, max_index
       


class Vertex():
    def __init__(self, coord, ngh, index):
        self.coord = coord
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

base_str='../Juelich-Receptor-Atlas/civet/mri1/surfaces/mri1_mid_surface_right_{}.obj'
base_mesh_fn=base_str.format(81920)
out_obj_fn='test.obj'
surf = Surface(base_mesh_fn)

surf.upsample()

surf.to_filename(out_obj_fn)

