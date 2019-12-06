""" Utility functions for particle environments
"""

import bisect
import numpy as np
from copy import deepcopy
from collections import defaultdict
from scipy.integrate import quad
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

class DefaultParameters():
    """ parameters to be used for generating objects to avoid hardcoding
    """

    # agent params
    agent_size = 0.025
    collision_termination_probability = 0.7
    max_communication_distance = 0.5

    # landmark params
    landmark_size = 0.1

class TriangularException(Exception):
    ''' Custom exception to test if coefficients define lower traingular matris
    Notes:
        empty class still useful for unit testing
    '''
    pass

class IntervalException(Exception):
    ''' Custom exception to test if interval formatted correcly
    Notes:
        empty class still useful for unit testing
    '''
    pass

class NodeExistenceException(Exception):
    ''' Custom exception to test if node is part of graph
    Notes:
        empty class still useful for unit testing
    '''
    pass

class NodeIntegerKeyException(Exception):
    ''' Custom exception to test if node has integer key
    Notes:
        empty class still useful for unit testing
    '''
    pass


def truncate_or_pad(l, n):
    return l[:n] + [0]*(n-len(l))

def check_2way_communicability(entity1, entity2):
    ''' Check if 2 way communication is possible betwen 2 entities
    '''
    comm_12 = False
    comm_21 = False

    # check if entity2 is within entity1 transmit range
    if hasattr(entity1, 'is_entity_transmittable') and callable(entity1.is_entity_transmittable):
        comm_12 = entity1.is_entity_transmittable(entity2)
    else:
        # entity1 assumed to have unlimited transmission range
        comm_12 = True

    # check if entity2 is within entity1 transmit range
    if hasattr(entity2, 'is_entity_transmittable') and callable(entity2.is_entity_transmittable):
        comm_21 = entity2.is_entity_transmittable(entity1)
    else:
        # entity2 assumed to have unlimited transmission range
        comm_21 = True

    return (comm_12 and comm_21)

def is_collision(agent1, agent2):
    if agent1 == agent2:
        return False
    else:
        dist = distance(agent1, agent2)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

def delta_pos(entity1, entity2):
    ''' position of entity1 with respect to entity2
    '''
    return entity1.state.p_pos - entity2.state.p_pos

def delta_vel(entity1, entity2):
    ''' velocity of entity1 with respect to entity2
    '''
    return entity1.state.p_vel - entity2.state.p_vel

def distance(entity1, entity2):
    d_pos = delta_pos(entity1, entity2)
    assert len(d_pos)==2
    return np.sqrt(d_pos[0]**2 + d_pos[1]**2)
    # return np.linalg.norm(d_pos)

def nearest_point_on_line_segment_2d(a, b, p):
    ''' find point on line segment ab that is closest to p
    Notes:
     - https://math.stackexchange.com/questions/2193720/find-a-point-on-a-line-segment-which-is-the-closest-to-other-point-not-on-the-li
     Except the above link is incorrect, t = - vu/vv
    '''
    if not(len(a) == len(b) == len(p) == 2):
        raise Exception('Must be 2-dimensional')



    # find line coordinate
    v = b - a
    u = a - p
    vu = float(np.dot(v,u))
    vv = float(np.dot(v,v))

    if np.isclose(vv, 0.0):
        # if vv=0, then a=b, just return a
        return a
        
    t = -vu/vv
    if t < 0 or t > 1:
        uu = float(np.dot(u,u))
        g0 = uu
        g1 = np.sqrt(vv) + 2*vu + uu
        t = 0.0 if g0 < g1 else 1.0

    # find 2D coordinates
    return (1-t)*a + t*b

def map_to_interval(x, intervals):
    '''map x into valid intervals in I
    Args:
        - x (float):  value to be mapped into I
        - intervals (2d list-like): intervals to map x 
    Notes:
        - assumes intervals in I are non-overlapping and in
        a sorted 2d array
    '''

    # ensure that we are working with an ndarray of floats
    I = deepcopy(intervals)
    if not isinstance(I, np.ndarray):
        I = np.array(I)
    I = intervals.astype(float)
    
    # check format of input
    I_shape = np.shape(I)
    if (I_shape[0] < 1 or 
        I_shape[1] != 2):
        raise IntervalException('Interval misformated. I={}'.format(I))

    # check I is properly sorted
    I_flat = np.squeeze(I.reshape(1, I_shape[0]*I_shape[1]))
    if (not all(I_flat[i] <= I_flat[i+1] for i in range(len(I_flat)-1))):
        raise IntervalException('Intervals not sorted. I={}'.format(I))

    # find lower and upper interval indices
    li = bisect.bisect_right(I[:,0], x)
    ui = bisect.bisect_left(I[:,1], x)

    # test intervals
    if li == ui+1:
        # if li=ui+1, then x is already inside a valid interval, return x
        return x

    elif li == ui:
        if li == 0:
            # x is below all intervals, map to global min
            return I[0,0]
        elif ui == len(I):
            # x is above all intervals, map to global max
            return I[-1,1]
        else:
            # x is in an inter-range, map to nearest interval
            lv = I[li-1,1]
            dl = x - lv
            uv = I[ui,0]
            du = uv - x

            if dl > du:
                return uv
            elif du > dl:
                return lv
            else:
                if np.random.rand() > 0.5:
                    return uv
                else:
                    return lv


    else:
        raise IntervalException('Interval misformated. I={}'.format(I))

def linear_index_to_lower_triangular(k):
    """ lower triangular matrix indices from linear index
    Note:
        get the indices of a lower triangular matrix (incl. main diagonal)
            from a linear index
        linear index is row-major, i.e. traverse left-to-right, then down;
            like reading a page of text.
    Args:
        k: linear index (0-indexed)
    Returns:
        ij: tuple of the element indices in lower triangular matrix (0-indexed)
    """

    # protect against unexpected behavior if used with non integer
    if not isinstance(k, int) or k < 0.0:
        raise Exception('linear index must be of type int and positive')

    # transform to 1-indexed
    k1 = k + 1

    # calculate row index (1-indexed) 
    i1 = np.ceil(0.5*(-1.0+np.sqrt(1 + 8*k1)))

    # calculate max index of previous row (1-indexed)
    K1_ = 0.5*i1*(i1-1)

    # calculate column index (1-indexed)
    j1 = k1 - K1_

    # double check that these are integers
    i = i1-1
    j = j1-1
    if not (i.is_integer() and j.is_integer()):
        raise Exception('Something went wrong, indices are non-integer')

    # transform into 0-indexed and return
    return (int(i), int(j))

# def lower_triangular_to_linear_index(ij):
#     """ get linear index from lower triangular index
#     Note:
#         get the linear index from a lower triangular matrix (incl. main diagonal)
#             indices
#         linear index is row-major, i.e. traverse left-to-right, then down;
#             like reading a page of text.
#     Args:
#         ij: tuple of the element indices in lower triangular matrix (0-indexed)
#     Returns:
#         k: linear index (0-indexed)
#     """
#     # protect against unexpected behavior if used with non integer
#     i = ij[0]
#     j = ij[1]
#     if not isinstance(i, int) or i < 0.0 not isinstance(j, int) or j < 0.0:
#         raise Exception('linear index must be of type int and positive')

#     k = i

def vertex_orientation_2d(vertex):
    ''' checks orientation of vertex defined by points pa->pb->pc
    Args:
     - vertex: 3-tuple of ordered 2d points that define vertex
    Returns:
     - (-1, 0, 1): negative orientation, non-orientable, positive orientation
    Notes:
     - https://en.wikipedia.org/wiki/Curve_orientation
    '''
    assert len(vertex) == 3
    pa = vertex[0]
    pb = vertex[1]
    pc = vertex[2]
    assert len(pa) == len(pb) == len(pc) == 2

    o_matrix = np.vstack((pa, pb, pc))
    o_matrix = np.concatenate((np.ones(shape=(3,1)), o_matrix), axis=1)
    assert o_matrix.shape == (3,3)

    # calculate determinate and set equal to zero if less than machine precision
    o_det = np.linalg.det(o_matrix)
    if abs(o_det) < np.finfo('float32').eps:
        return 0, o_det
    else:
        return np.sign(o_det), o_det
 
class UndirectedGraph(object):
    ''' Undirected Graph
    '''
 
    def __init__(self):
 
        # default dictionary to store graph
        self.graph = defaultdict(list)
 
    def add_edge(self,u,v):
        ''' add an edge to graph (implicitly adds nodes)'''
        self.graph[u].append(v)
        self.graph[v].append(u)

    def add_node(self,u):
        ''' explicitly add node to graph without any edges '''
        self.graph[u]
 
    def breadth_first_connectivity_search(self, start_node, end_node):
        ''' breadth first search to determine connectivity of two nodes
        '''

        # check that query nodes are part of graph
        if not (start_node in self.graph.keys() and 
            end_node in self.graph.keys()):
            raise NodeExistenceException()

        # return true if start_node == end_node
        # Note: this does not imply there is a loop edge between a node
        # and itself. It just means that the node is reachable by itself
        if start_node == end_node:
            return True
 
        # Mark all the vertices as not visited
        # visited = [False] * (len(self.graph))
        visited = {k: False for k in self.graph.keys()}
 
        # Create a queue for BFS
        queue = []
 
        # Mark the source node as 
        # visited and enqueue it
        s = start_node
        queue.append(s)
        visited[s] = True
 
        while queue:
 
            # Dequeue a vertex
            s = queue.pop(0)
 
            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.graph[s]:
                if i == end_node:
                    # once the end node is found, we know it's connected, return True
                    return True
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

        # if end node was never encountered, then it's not connected to start node
        return False

class SimpleNetwork(UndirectedGraph):
    ''' simple network that only stores connectivity between nodes (no weights, directions)
    Args:
     - nodes [list] list-like data structure containing node objects.
    '''
    def __init__(self, nodes=None):
        UndirectedGraph.__init__(self)
        self.nodes = nodes
        self.n_nodes = len(nodes)

        # since nodes maybe unhashable, add to graph using index in nodes list
        for k in range(self.n_nodes):
            self.add_node(k)

    def add_edge(self,u,v):
        ''' add an edge to graph if nodes already exist (does not implicitly add node)'''
        
        if not (isinstance(u, int) and isinstance(v, int)):
            raise NodeIntegerKeyException()

        if u not in self.graph:
            raise NodeExistenceException()

        if v not in self.graph:
            raise NodeExistenceException()

        UndirectedGraph.add_edge(self,u,v)


class ResistanceNetwork(UndirectedGraph):
    ''' object that describes a connected network of resistors
    References:
     - Wu, F. Y. (2004). Theory of resistor networks: the two-point resistance. 
     Journal of Physics A: Mathematical and General, 37(26), 6653.
    Notes:
     - WARNING: There is no check for terminated nodes because this object is
     created at every time step. There is no guarantee that this class won't
     be used differently elsewhere, though, so this is likely unsafe 
    '''
    def __init__(self, n_nodes=None, resistance_array=None):
        '''
        Args:
         - resistance_array: array of direct, immediate resistances between nodes
        '''

        # instantiate graph object
        UndirectedGraph.__init__(self)
        self.resistance_array = resistance_array
        self.n_nodes = n_nodes

        # check valid number of nodes
        if not isinstance(self.n_nodes, int) or self.n_nodes < 0:
            raise Exception('invalid number of nodes')

        # check validaty for resistance array
        if not self.n_nodes == 0.5*(-1 + np.sqrt(1+8*len(self.resistance_array))):
            raise TriangularException('resistance_array list inproperly sized')

        # fill in unweighted UndirectedGraph object and laplacian matrix
        L = np.array([[0.0]*self.n_nodes]*self.n_nodes)
        for k, rij in enumerate(self.resistance_array):
            i, j = linear_index_to_lower_triangular(k)
            if i == j:
                # leave placeholder of zero and do nothing
                pass
            else:
                L[i][j] = L[j][i] = -float(1.0/rij)

            if not np.isinf(rij):
                self.add_edge(i,j)

        # calculate diagonal entries
        for i in range(len(L)):
            L[i][i] = -np.sum(L[i])

        # store laplacian matrix
        self.laplacian_matrix = L

        # calculate nonzero eigenpairs
        self.eigenpairs = self.get_nonzero_eigenpairs()


    # def get_laplacian_matrix(self):
    #     ''' calculate the laplacian matrix from the resistance vector
    #     Reference:
    #      - Wu (2004): Section I; Equation 4, 5
    #     '''
    #     # initilize 2D array
    #     L = np.array([[0.0]*self.n_nodes]*self.n_nodes)

    #     # calculate conductance
    #     for k, rij in enumerate(self.resistance_array):
    #         i, j = linear_index_to_lower_triangular(k)
    #         if i == j:
    #             # leave placeholder of zero and do nothing
    #             pass
    #         else:
    #             L[i][j] = L[j][i] = -float(1.0/rij)

    #     # calculate diagonal entries
    #     for i in range(len(L)):
    #         L[i][i] = -np.sum(L[i])

    #     return L

    def get_nonzero_eigenpairs(self):
        ''' extract the eigenpairs (eigenvalue, eigenvector) for non-zero eigenvalues of laplacian
        '''
        vals, vecs = np.linalg.eig(self.laplacian_matrix)
        eigenpairs = []
        for i, val in enumerate(vals):

            if not np.isclose(val, 0):
                eigenpairs.append([val, vecs[:,i]])

        return eigenpairs

    def get_two_point_resistance(self, a, b):
        ''' calculate resistance between two nodes in network
        Args:
         - a, b: index of nodes to calculate resistance
        '''
        if self.breadth_first_connectivity_search(a,b):
            Rab = 0.0
            for i, p in enumerate(self.eigenpairs):
                # calculate resistance
                # Reference Wu (2004): Section I; Equation 4, 5
                Rab += (1.0/p[0])*(p[1][a] - p[1][b])**2

        else:
            Rab = np.inf

        # assert that resistance is real
        if isinstance(Rab, complex):
            assert np.isclose(Rab.imag, 0.0)
            Rab = Rab.real

        # assert resistance is positive
        assert not Rab < 0      

        return Rab



class MultivariatePolynomial:
    def __init__(self, coefs=None):
        self.coefs = coefs

class MVP2D(MultivariatePolynomial):
    """ 2-dimensional multivariate polynomial
    """
    def __init__(self, coefs=None):
        MultivariatePolynomial.__init__(self, coefs)

    def term_index_to_exponents(self,k):
        """ return the x and y exponents for the kth term of the polynomial

        For the term in the polynomial:  a_k * x**p * y**q
        find p and q from k (k is 0-indexed)

        Args:
            k: index of term in polynomial (k = 0,1,2,....)
        Returns:
            (p,q): tuple of the x and y exponents, respectively
        """
        ij = linear_index_to_lower_triangular(k)
        i = ij[0]
        j = ij[1]
        p = i - j
        q = j
        return (p,q)

    def evaluate(self,x,y):
        """ evaluate the polynomial at position x,y
        """

        # check if coefficients list is appropriate length
        Kt = len(self.coefs)
        rt = 0.5*(-1 + np.sqrt(1+8*Kt))
        if not rt.is_integer():
            raise TriangularException('coefficient list inproperly sized')

        val = 0
        # incrementally add polynomial terms
        for k, a in enumerate(self.coefs):
            exp = self.term_index_to_exponents(k)
            val = val + a * x**exp[0] * y**exp[1]

        return val

    def convex_hull_integral(self, points):
        """ calculate double integral over convex hull of set of points
        Notes:
         - use scipy to generate convex hull with positive oriented contour
         - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
        """
        try:
            hull = ConvexHull(points)
        except QhullError:
            # if hull is degenerate (e.g. all points collinear), then set integral to zero
            return 0.0

        # perform line integral over each boundary segment
        I = 0.0
        for i, _ in enumerate(hull.vertices[:-1]):
            p1 = points[hull.vertices[i]]
            p2 = points[hull.vertices[i+1]]
            I += self.augmented_line_integral(p1, p2)
        I += self.augmented_line_integral(p2, points[hull.vertices[0]])

        return I

    def augmented_line_integral(self, p1, p2):
        """ line integral of M*dy from points p1 to p2 where M such that delM/delx = MVP
        Args:
            p1: 2-tuple of x,y-coords of first point
            p2: 2-tuple of x,y-coords of second point
        Notes:
         - This can be used as a subfunction to calculate the double integral over
         an area defined by piecewise boundaries.
         - Formatting based on Green's theorem and assuming:
            L = 0
            M = sum over k of: a_k/(i-j+1) * x**(p+1) * y**(q)
         - line parameterized with respect to t from 0 <= t <= 1
         - x = (x2-x1)*t + x1
         - y = (y2-x1)*t + y1
         - dy = (y2-y1)*dt
        """

        # extract coords
        assert len(p1) == 2
        assert len(p2) == 2
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]

        def integrand(t, p, q):
            return ((x2-x1)*t + x1)**(p+1) * ((y2-y1)*t + y1)**q

        I_C = 0.0
        for k, ak in enumerate(self.coefs):
            ij = linear_index_to_lower_triangular(k)
            i = ij[0]
            j = ij[1]
            p = i - j
            q = j
            I = quad(integrand, 0.0, 1.0, args=(p,q))

            # check that there is no imaginary component
            if not np.isclose(I[1], 0.0, atol=1e-8):
                if np.isclose(I[0], 0.0) or (not np.isclose(I[0], 0.0) and not np.isclose(I[0]/I[1],0, atol=1e-6)):
                    if not np.isclose(I[1]/I[0], 0.0):
                        print("Integral has non-zero imaginary component. \n I_real={}, I_imag={}".format(I[0], I[1]))
                        print("k={}, a_k={} i={}, j={}, p={}, q={} \n p1={} \n p2={} \n coefs={} \n".format(k,ak,i,j,p,q,p1,p2,self.coefs))
                        raise Exception
            I_C += ak/float(i+1.0-j) * I[0]

        return (y2-y1) * I_C 


class RewardFunction(object):
    ''' function for reward distribution throughout world
    Attributes:
    Notes:
    '''
    def __init__(self):
        pass

    def get_value(self, *args, **kwargs):
        ''' return the reward value at given state
        '''
        raise NotImplementedError('Child class must override get_value')


class RewardFunction2D(RewardFunction):
    """ Probabilistic, parameterized reward function

    Attributes:
        bounds (dict): dictionry of spatial bounds outside which function evals to 0
    """
    def __init__(self, bounds):
        self.spatial_bounds = bounds


    def get_value(self,x,y):
        ''' return the reward value at position x,y
        Args:
            - x, y [m]: absolute position for reward evaluation
        '''
        raise NotImplementedError('Child class must override get_value')

    def get_convex_hull_integral(self, points):
        ''' surface integrall of reward function over convex hull defined by set of points
        Args:
         - points: list-like of 2D points
        Notes:
         - Special care is needed to handle bounds conditions for each child class
        '''
        raise NotImplementedError('Child class must override get_convex_hull_integral')

    def check_in_bounds(self,x,y):
        """ check if coordinates x,y are in bounds of reward function
        """

        if self.spatial_bounds is None:
            return True

        eps = np.finfo(np.float32).eps
        if (x < self.spatial_bounds['xmin']-eps or 
            x > self.spatial_bounds['xmax']+eps or
            y < self.spatial_bounds['ymin']-eps or 
            y > self.spatial_bounds['ymax']+eps):
            return False
        else:
            return True

class PolynomialRewardFunction2D(RewardFunction2D):
    """ Reward function based on a multivariate polynomial
    Attributes:
        - mvp2d (MVP2D): 2D multivariate polynomial describing reward
    Notes:
        - mvp2d has form:
            c_0 + c_1*x + c_2*y + c_3*x^2 + c_4*x*y + c_5*y^2 + ... = 
            c_0*x^0*y^0 + c_1*x^1*y^0 + c_2*x^0*y^1 + c_3*x^2*y^0 + ...
            c_4*x^1*y^1 + c_5*x^0*y^2 + c_6*x^3*y^0 + c_7*x^2*y^1 + ...
            c_8*x^1*y^2 + c_9*x^0*y^3 + ...
    """

    def __init__(self, coefs=None, bounds=None):
        RewardFunction2D.__init__(self, bounds)
        self.mvp2d = MVP2D(coefs)

    def get_value(self,x,y):
        ''' return the reward value at position x,y
        Args:
            - x, y [m]: absolute position for reward evaluation
        Notes:
        '''

        if self.check_in_bounds(x,y):
            return self.mvp2d.evaluate(x,y)
        else:
            return 0.0

class RadialPolynomialRewardFunction2D(PolynomialRewardFunction2D):
    ''' Polynomial Reward Function of specific shape parameterized by a radius and max
    Notes:
     - Shaped such that reward is max at 0,0 and drops off smoothly
     to 0 at r
    '''
    def __init__(self, radius, peak_value, bounds=None, is_cost=False):
        '''
        Args:
         - radius [m]: spatial radius of risk function
         - peak_value: max reward or cost
         - is_cost: flips function to be all negative
        '''
        assert(radius > 0)
        assert(not peak_value < 0)
        if bounds is None:
            bounds = {'xmin': -radius, 'xmax': radius, 'ymin':-radius, 'ymax':radius}
        r2 = float(radius*radius)
        coef_arr = peak_value*np.array([1.0, 0.0, 0.0, -1.0/r2, 0.0, -1.0/r2])
        if is_cost:
            coef_arr = -coef_arr
        PolynomialRewardFunction2D.__init__(self, coefs=list(coef_arr), bounds=bounds)

        # store values for later reference
        self.radius = radius
        self.peak_value = peak_value

    def get_value(self,x,y):
        ''' return the reward value at position x,y
        Args:
            - x, y [m]: absolute position for reward evaluation
        Notes:
            - Same as PolynomialRewardFunction2D.get_value except it checks 
            radius instead of cartesian bounds
        '''

        if x**2 + y**2 < self.radius**2:
            return self.mvp2d.evaluate(x,y)
        else:
            return 0.0

class ExtendedRadialPolynomialRewardFunction2D(RadialPolynomialRewardFunction2D):
    ''' Polynomial Reward Function of specific shape parameterized by a radius and peak_value
    Notes:
     - same as RadialPolynomialRewardFunction2D except it is goes negative outside of the 
     radius instead of being clipped at zero
    '''
    def __init__(self, radius, peak_value, is_cost=False):
        '''
        Args:
         - radius [m]: spatial radius of risk function
         - peak_value: max reward or cost
         - is_cost: flips function to be all negative
        '''
        bounds = {'xmin': -np.inf, 'xmax': np.inf, 'ymin':-np.inf, 'ymax':np.inf}
        RadialPolynomialRewardFunction2D.__init__(self, 
                                                  radius=radius, 
                                                  peak_value=peak_value,
                                                  bounds=bounds, 
                                                  is_cost=is_cost)


    def get_value(self,x,y):
        ''' return the reward value at position x,y
        Args:
            - x, y [m]: absolute position for reward evaluation
        Notes:
            - Same as PolynomialRewardFunction2D.get_value except it checks 
            radius instead of cartesian bounds
        '''
        return self.mvp2d.evaluate(x,y)

    def get_convex_hull_integral(self, points):
        ''' surface integrall of reward function over convex hull defined by set of points
        Args:
         - points: list-like of 2D points
        Notes:
         - No need to worry about intersecting convex hull with reward boundaries
         since they are infinite for ExtendedRadialPolynomialRewardFunction2D
        '''
        return self.mvp2d.convex_hull_integral(points)

    def get_radial_integral(self, r):
        '''calculate integral of reward function over  specified radius
        Notes:
         - Reward function is a parabolic curve, not circular. To calc 
         "volume" of reward function, take double integral from 0->2*pi, 0->r
         of peak_value(1- x**2/radius**2 - y**2/radius**2) dx dy =
         peak_value * r * (1 - r**2/R**2) dr dtheta
        '''
        # return 2.0*np.pi*self.radius*(r**2/2.0 - r**4/(4.0*self.radius**2))
        P = float(self.peak_value)
        R = float(self.radius)
        r = float(r)
        # return np.pi * P**2 * r / (R**4) * (R**4 - 2.0*R**2*r**2/3.0 + r**4/5.0)
        # return 2.0 * np.pi * P/R**2 * (0.5*R**2*r**2 - 0.25*r**4)
        return np.pi * P * r**2/R**2 * (R**2 - 0.5 * r**2)


class UniformRewardFunction2D(RewardFunction2D):
    """ Reward function based on a uniform random variable parameterized in 2D

    Probabilistic function for drawing a uniform random variable where the limits
    of the uniform distirubution are a function of two dimensional position x,y

    Attributes:
        lower_mvp2d (MVP2D): 2D multivariate polynomial to describe lower limit of distribution
            as a function of spatial position
        upper_mvp2d (MVP2D): 2D multivariate polynomial to describe upper limit of distribution
            as a function of spatial position 

    Notes:
        - lower and upper limit of the distribution should not be confused with the 
        spatial bounds of the reward function
        - The distribution of reward is uniform at a specific point x,y, but that
        is not to say that uniform distribution varies across the spatial bounds
        of the reward function
    """
    def __init__(self, configs):
        RewardFunction2D.__init__(self, configs['bounds'])
        self.lower_mvp2d = MVP2D(configs['coefficients']['lower'])
        self.upper_mvp2d = MVP2D(configs['coefficients']['upper'])

    def get_distribution_limits(self,x,y):
        """ get lower and upper limits of distribution at coords x,y

        Note: it is difficult to guarantee that the coofficients used for
            strictly evaluate to lower(x,y) <= upper(x,y). Instead of
            enforcing this, just swap the limits in the event that "lower"
            is greater than upper  
        """
        lower = self.lower_mvp2d.evaluate(x,y)
        upper = self.upper_mvp2d.evaluate(x,y)

        if lower > upper:
            return (upper, lower)
        else:
            return (lower, upper)

    def get_value(self,x,y):
        ''' return the reward value at position x,y
        Args:
            - x, y [m]: absolute position for reward evaluation
        Notes:
            - this just bounces the call to draw_random_sample but is used
            to be compatible with parent definitions
        '''
        return draw_random_sample(x,y)

    def draw_random_sample(self,x,y):
        """ draw a random sample of reward function at spatial coords x,y
        Args:
            x (float): x-coordinate at which to draw random sample
            y (float): y-coordinate at which to draw random sample 
        Returns:
            r (float): random sample of reward function
        """

        # return 0.0 if x,y outside of function's spatial bounds
        if not self.check_in_bounds(x,y):
            return 0.0

        # get lower and upper bounds of distribution at spatial position x,y
        dist_limits = self.get_distribution_limits(x,y)

        # draw random sample
        return np.random.uniform(dist_limits[0], dist_limits[1])

    def expected_value(self,x,y):
        """ calculate expected value of distribution at coords x,y
        Args:
            x (float): x-coordinate at which to draw random sample
            y (float): y-coordinate at which to draw random sample 
        Returns:
            rx (float): expected value of reward function at x,y
        """

        # return 0.0 if x,y outside of function's spatial bounds
        if not self.check_in_bounds(x,y):
            return 0.0

        # get lower and upper bounds of distribution at spatial position x,y
        dist_limits = self.get_distribution_limits(x,y)

        # calculate expected value of uniform distribution
        return 0.5*(dist_limits[1] + dist_limits[0])

class RiskFunction(object):
    ''' function for risk distribution throughout world
    Attributes:
    Notes:
    '''
    def __init__(self):
        pass

    def sample_failure(self, *args, **kwargs):
        ''' return boolean describing if risk function has caused a agent to fail 
        '''
        raise NotImplementedError('Child class must override sample_failure')

    def get_failure_probability(self, *args, **kwargs):
        ''' return probability of agent failure  
        '''
        raise NotImplementedError('Child class must override get_failure_probability')

class NoneRiskFunction(RiskFunction):
    ''' function for risk distribution throughout world
    Attributes:
    Notes:
    '''
    def __init__(self):
        pass

    def sample_failure(self, *args, **kwargs):
        ''' return boolean describing if risk function has caused a agent to fail 
        '''
        return False

    def get_failure_probability(self, *args, **kwargs):
        ''' return probability of agent failure  
        '''
        return 0.0

class BernoulliRiskFunction2D(RiskFunction):
    """ Risk function in 2D where a multivariate polinomial discribes probability of failure

    Attributes:
        bounds (dict): dictionry of spatial bounds outside which function evals to 0
        mvp2d (MVP2D): multivariate polynomial describing probability of failure at x,y 

    Notes:
        - if mvp2d evaluates outside of [0,1], it will be mapped to 0 or 1
    """
    def __init__(self, coefs, bounds):
        self.spatial_bounds = bounds
        self.mvp2d = MVP2D(coefs)


    def sample_failure(self,x,y):
        ''' return boolean describing if risk function has caused a agent to fail 
        Args:
            - x, y [m]: absolute position for failure evaluation
        '''
        if self.check_in_bounds(x,y):
            p_xy = self.get_failure_probability(x,y)
            fail = np.random.binomial(1, p_xy)
            return bool(fail)
        else:
            return False

    def get_failure_probability(self,x,y):
        ''' get probability of risk function causing failure at specific location
        '''
        if self.check_in_bounds(x,y):
            p_xy = self.mvp2d.evaluate(x,y)
            p_xy = min(max(p_xy,0.0),1.0)
            return p_xy
        else:
            return 0

    def check_in_bounds(self,x,y):
        ''' check if coordinates x,y are in bounds of risk function
        Args:
            - x, y [m]: absolute position for failure evaluation
        '''

        if self.spatial_bounds is None:
            return True

        eps = np.finfo(np.float32).eps
        if (x < self.spatial_bounds['xmin']-eps or 
            x > self.spatial_bounds['xmax']+eps or
            y < self.spatial_bounds['ymin']-eps or 
            y > self.spatial_bounds['ymax']+eps):
            return False
        else:
            return True

class RadialBernoulliRiskFunction2D(BernoulliRiskFunction2D):
    ''' Bernoulli Risk Function of specific shape parameterized by a radius
    Notes:
     - Shaped such that failure probability is 1 at 0,0 and drops off smoothly
     to 0 at r
    '''
    def __init__(self, radius, peak_value=1.0):
        '''
        Args:
         - radius [m]: spatial radius of risk function
        '''
        assert(radius > 0)
        assert peak_value > 0 and peak_value <= 1.0
        bounds = {'xmin': -radius, 'xmax': radius, 'ymin':-radius, 'ymax':radius}
        r2 = float(radius*radius)
        coefs = peak_value*np.array([1.0, 0.0, 0.0, -1.0/r2, 0.0, -1.0/r2])
        BernoulliRiskFunction2D.__init__(self, coefs=coefs, bounds=bounds)