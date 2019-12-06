#!/usr/bin/env python

# suite of unit tests for particle_environments/common.py. To run test, simply call:
#
#   in a shell with conda environment ergo_particle_gym activated:
#   nosetests test_particle_environments_common.py
#
#   in ipython:
#   run test_particle_environments_common.py

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'particle_environments/'))

import unittest
import numpy as np
import common as tl
import random


class TestTruthFunctions(unittest.TestCase):
    ''' test functions in the truth_library
    '''

    def test_map_to_interval_1(self):

        # trial 1
        x = 0.0
        I = np.array([[-1, 1]])
        x_mapped = tl.map_to_interval(x,I)
        self.assertAlmostEqual(x_mapped, 0.0)

        # trial 2
        x = 1.5
        I = np.array([[-1, 2]])
        self.assertAlmostEqual(tl.map_to_interval(x,I), x)

        # trial 3
        x = 2.5
        I = np.array([[-1, 2]])
        self.assertAlmostEqual(tl.map_to_interval(x,I), 2.0)

        # trial 4
        x = -8
        I = np.array([[34, 80]])
        self.assertAlmostEqual(tl.map_to_interval(x,I), 34.0)

        # trial 5
        x = 46
        I = np.array([[-67, -63]])
        self.assertAlmostEqual(tl.map_to_interval(x,I), -63)

        # trial 6
        x = -163.82185004073605
        I = np.array([[  -8.43082759,  105.75927842]])
        self.assertAlmostEqual(tl.map_to_interval(x,I), -8.43082759)

        # trial 6
        x = 99.09556113970952
        I = np.array([[ -30.26691791,  117.50824726]])
        self.assertAlmostEqual(tl.map_to_interval(x,I), 99.09556113970952)

    def test_map_to_interval_2(self):

        # trial 1
        x = 11.210647963197607
        I = np.array([[ -89.23979646,  -76.19363027],
                      [ -17.81051723,   -5.89888779],
                      [  11.61859015,   46.39421969],
                      [  98.27102042,  136.46498318]])
        self.assertAlmostEqual(tl.map_to_interval(x,I), 11.61859015)

        # trial 2
        x = 23.057600176362314
        I = np.array([[-115.2940264 ,  -18.95484345],
                      [  -3.75983112,   31.98463023],
                      [  68.67521893,   82.87224541],
                      [  89.32899299,  101.97467692],
                      [ 120.39250067,  153.19466243]])
        self.assertAlmostEqual(tl.map_to_interval(x,I), 23.057600176362314)

    def test_map_to_interval_exceptions_1(self):

        x = 86.33052678061554
        I = np.array([[  36.38999474,  129.6911174 ],
                      [  33.35214228,  -20.64645801]])
        with self.assertRaises(tl.IntervalException):
                tl.map_to_interval(x,I)

    def test_map_to_interval_exceptions_2(self):

        x = 86.33052678061554
        I = np.array([[  36.38999474,  129.6911174, 0.1 ],
                      [  33.35214228,  -20.64645801, 10]])
        with self.assertRaises(tl.IntervalException):
                tl.map_to_interval(x,I)

    def test_nearest_point_on_line_segment_2d_1(self):
        a = np.array([-1,0])
        b = np.array([1,0])
        plist = [np.array([0,1]), np.array([-1,1]), np.array([1,1]), np.array([-0.43178172,0]), np.array([4.33210372, 3.61065622])]
        exp = [np.array([0,0]), np.array([-1,0]), np.array([1,0]), np.array([-0.43178172,0]), np.array([1,0])]
        for i, p in enumerate(plist):
            res = tl.nearest_point_on_line_segment_2d(a,b,p)
            self.assertAlmostEqual(res[0], exp[i][0])
            self.assertAlmostEqual(res[1], exp[i][1])

    def test_nearest_point_on_line_segment_2d_2(self):
        a = np.array([-4.78902451, -0.46995467])
        b = np.array([1.28928092, 2.94187194])
        plist = [np.array([0.43960779, 3.223054  ]), np.array([-6.25526615e+00,  4.70240880e-04]), np.array([4.54793307, 7.78450475])]
        exp = [np.array([0.76319369, 2.64657279]), a, b]
        for i, p in enumerate(plist):
            res = tl.nearest_point_on_line_segment_2d(a,b,p)
            self.assertAlmostEqual(res[0], exp[i][0])
            self.assertAlmostEqual(res[1], exp[i][1])

    def test_nearest_point_on_line_segment_2d_3(self):
        a = np.array([ 7.33567341, -9.91835693])
        b = np.array([ 7.33567341, -9.91835693])
        plist = [np.array([4.74649332, 3.35905451]), np.array([-8.10593068, -6.11380121]), np.array([ 2.18299351, -1.48674994])]
        exp = [b, b, b]
        for i, p in enumerate(plist):
            res = tl.nearest_point_on_line_segment_2d(a,b,p)
            self.assertAlmostEqual(res[0], exp[i][0])
            self.assertAlmostEqual(res[1], exp[i][1])

    def test_vertex_orientation_2d_1(self):
        '''vertex_orientation_2d: known vertices with positive, non, and neg orientation'''
        p1 = [0,0]
        p2 = [1,0]
        p3 = [1,1]
        self.assertEqual(tl.vertex_orientation_2d((p1,p2,p3))[0], 1.0)

        p1 = [0,0]
        p2 = [1,0]
        p3 = [2,0]
        self.assertEqual(tl.vertex_orientation_2d((p1,p2,p3))[0], 0)

        p1 = [0,0]
        p2 = [1,0]
        p3 = [1,-1]
        self.assertEqual(tl.vertex_orientation_2d((p1,p2,p3))[0], -1.0)

    def test_vertex_orientation_2d_2(self):
        '''vertex_orientation_2d: random vertices with positive orientation'''
        for trial in range(100):
            p1 = np.array([np.random.uniform(-100, 100), 0])
            p2 = np.array([p1[0] + np.random.uniform(0, 200), 0])
            p3 = np.array([np.random.uniform(-100,100), np.random.uniform(1,100)])
            self.assertEqual(tl.vertex_orientation_2d((p1,p2,p3))[0], 1.0)

    def test_vertex_orientation_2d_3(self):
        '''vertex_orientation_2d: random vertices with non-orientation'''
        for trial in range(100):
            p1 = np.array([np.random.uniform(-100, 100), np.random.uniform(-100,100)])
            p2 = np.array([np.random.uniform(-100, 100), np.random.uniform(-100,100)])
            v12 = np.array([p2[0]-p1[0], p2[1] - p1[1]])
            l12 = np.linalg.norm(v12)
            l13 = np.random.uniform(-100,100)
            p3 = np.array([p1[0] + l13*v12[0]/l12, p1[1] + l13*v12[1]/l12])
            self.assertEqual(tl.vertex_orientation_2d((p1,p2,p3))[0], 0)

    def test_vertex_orientation_2d_4(self):
        '''vertex_orientation_2d: random vertices with random orientation'''
        for trial in range(100):
            p1 = np.array([np.random.uniform(-100, 100), np.random.uniform(-100,100)])
            p2 = np.array([np.random.uniform(-100, 100), np.random.uniform(-100,100)])
            p3 = np.array([np.random.uniform(-100, 100), np.random.uniform(-100,100)])
            v12 = np.array([p2[0]-p1[0], p2[1] - p1[1]])
            v23 = np.array([p3[0]-p2[0], p3[1] - p2[1]])
            exp_or = np.sign(np.cross(np.append(v12,0),np.append(v23,0))[2])
            self.assertEqual(tl.vertex_orientation_2d((p1,p2,p3))[0], exp_or)


class TestMVP2D(unittest.TestCase):
    ''' test the multivariate polynomial 2D class
    '''

    def test_evaluate_1(self):
        # polynomial = x^2 + y^2
        c = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        p = tl.MVP2D(c)
        self.assertAlmostEqual(p.evaluate(0.0,0.0), 0.0)
        self.assertAlmostEqual(p.evaluate(1.0,0.0), 1.0)
        self.assertAlmostEqual(p.evaluate(-1.0,1.0), 2.0)
        self.assertAlmostEqual(p.evaluate(0.0,1.0), 1.0)
        self.assertAlmostEqual(p.evaluate(-42.6733,-27.0083), 2550.4588017799997)

    def test_evaluate_2(self):
        # polynomial = 1 + x + y + x^2 + x*y + y^2
        c = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        p = tl.MVP2D(c)
        self.assertAlmostEqual(p.evaluate(0.0,0.0), 1.0)
        self.assertAlmostEqual(p.evaluate(1.0,0.0), 3.0)
        self.assertAlmostEqual(p.evaluate(-1.0,1.0), 2.0)
        self.assertAlmostEqual(p.evaluate(0.0,1.0), 3.0)
        self.assertAlmostEqual(p.evaluate(-42.6733,-27.0083), 3634.310490169999)

    def test_evaluate_3(self):
        # polynomial = 1
        c = [1.0]
        p = tl.MVP2D(c)
        self.assertAlmostEqual(p.evaluate(0.0,0.0), 1.0)
        self.assertAlmostEqual(p.evaluate(1.0,0.0), 1.0)
        self.assertAlmostEqual(p.evaluate(-1.0,1.0), 1.0)
        self.assertAlmostEqual(p.evaluate(0.0,1.0), 1.0)
        self.assertAlmostEqual(p.evaluate(-42.6733,-27.0083), 1.0)

    def test_evaluate_4(self):
        # polynomial = (randomly generated coefs)
        c = [4.9888785066963,
            -2.6072258560952,
            2.6480911731013,
            1.9870246909498,
            4.6454780151348,
            2.2809308451977,
            -9.5205873062464,
            0.850583925308,
            9.0531745362343,
            -9.9005566723182]
        p = tl.MVP2D(c)
        self.assertAlmostEqual(p.evaluate(0.0,0.0), 4.9888785066963)
        self.assertAlmostEqual(p.evaluate(1.0,0.0), -5.1519099646955)
        self.assertAlmostEqual(p.evaluate(-1.0,1.0), 1.2841130799074048)
        self.assertAlmostEqual(p.evaluate(0.0,1.0), 0.017343852677100813)
        self.assertAlmostEqual(p.evaluate(-42.6733,-27.0083), 621923.6139074472)

    def test_evaluate_failure(self):
        c = [1.0, 1.0]
        p = tl.MVP2D(c)
        with self.assertRaises(tl.TriangularException):
            p.evaluate(0.0,0.0)

        c = [1.0, 1.0, 1.0, 0.0]
        p = tl.MVP2D(c)
        with self.assertRaises(tl.TriangularException):
            p.evaluate(-42.6733,-27.0083)

        c = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        p = tl.MVP2D(c)
        with self.assertRaises(tl.TriangularException):
            p.evaluate(-42.6733,-27.0083)

    def test_augmented_line_integral_1(self):
        '''MVP2D.augmented_line_integral: test integral of unit cube'''
        c = [1.0]
        mvp = tl.MVP2D(c)

        # test full cube
        verts = [(0,0), (1,0), (1,1), (0,1)]
        I = 0.0
        for i,_ in enumerate(verts[:-1]):
            I += mvp.augmented_line_integral(verts[i], verts[i+1])

        I += mvp.augmented_line_integral(verts[-1], verts[0])
        self.assertAlmostEqual(I, 1.0)

        # test half cube
        verts = [(0,0), (1,0), (1,1)]
        I = 0.0
        for i,_ in enumerate(verts[:-1]):
            I += mvp.augmented_line_integral(verts[i], verts[i+1])

        I += mvp.augmented_line_integral(verts[-1], verts[0])
        self.assertAlmostEqual(I, 0.5)

    def test_augmented_line_integral_2(self):
        '''MVP2D.augmented_line_integral: square region under parabolic dome'''
        coefs = np.array([1.0, 0.0, 0.0, -1.0, 0.0, -1.0])
        mvp = tl.MVP2D(coefs)
        b = np.sqrt(2.0)/2.0
        verts = [(-b,-b), (b,-b), (b,b), (-b,b)]
        expected_integral = 4.0*b**2 - 8.0/3.0 * b**4

        I = 0.0
        for i,_ in enumerate(verts[:-1]):
            I += mvp.augmented_line_integral(verts[i], verts[i+1])
        I += mvp.augmented_line_integral(verts[-1], verts[0])

        self.assertAlmostEqual(I, expected_integral)

    def test_augmented_line_integral_3(self):
        '''MVP2D.augmented_line_integral: ramp through origin integrates to zero over symmetric bounds'''
        for trial in range(100):

            # generate coefs of 2D ramp through origin
            a1 = np.random.uniform(-10, 10)
            a2 = np.random.uniform(-10, 10)
            coefs = np.array([0.0, a1, a2])
            mvp = tl.MVP2D(coefs)

            # form rectangular region centered on zero
            xc = np.random.uniform(0, 10)
            yc = np.random.uniform(0, 10)
            verts = [(-xc,-yc), (xc,-yc), (xc,yc), (-xc,yc)]

            # perform integral and test it is approximately 0
            I = 0.0
            for i,_ in enumerate(verts[:-1]):
                I += mvp.augmented_line_integral(verts[i], verts[i+1])
            I += mvp.augmented_line_integral(verts[-1], verts[0])

            self.assertAlmostEqual(I, 0.0)

    def test_augmented_line_integral_4(self):
        '''MVP2D.augmented_line_integral: parabollic prism with pos and neg integrate to zero'''
        for trial in range(100):

            # generate coefs of parabolic prism that ensure positve and negative regions
            a0 = np.random.uniform(-10, 10)
            a3 = np.random.uniform(-10, 10)
            a3 = -a3 if np.sign(a3)==np.sign(a0) else a3

            # randomly assign prism along x or y axis
            if np.random.rand() > 0.5:
                coefs = np.array([a0, 0.0, 0.0, a3, 0.0, 0])
            else:
                coefs = np.array([a0, 0.0, 0.0, a3, 0.0, 0])
            mvp = tl.MVP2D(coefs)

            # form square region of integration
            b = np.sqrt(-3.0*a0/a3)
            verts = [(-b,-b), (b,-b), (b,b), (-b,b)]

            # perform integral and test it is approximately 0
            I = 0.0
            for i,_ in enumerate(verts[:-1]):
                I += mvp.augmented_line_integral(verts[i], verts[i+1])
            I += mvp.augmented_line_integral(verts[-1], verts[0])

            self.assertAlmostEqual(I, 0.0)

    def test_augmented_line_integral_5(self):
        '''MVP2D.augmented_line_integral: random triangular region under uniform curve with random length coeffs'''
        for trial in range(100):

            # generate random heigh of uniform curce
            a0 = np.random.uniform(-10, 10)

            # generate random length coefficient mvp
            coef_len = random.choice([1,3,6,10,15,21])
            coefs = np.concatenate(([a0], np.zeros(coef_len-1)))
            mvp = tl.MVP2D(coefs)

            # form random triangular path with positive orientation
            p1 = np.random.uniform(-10, 10, 2)
            p2 = np.random.uniform(-10, 10, 2)
            p3 = np.random.uniform(-10, 10, 2)
            verts = [p1, p2, p3]
            if tl.vertex_orientation_2d(verts)[0] < 0:
                verts = [p1, p3, p2]


            # find area of triangle region
            v12 = p2 - p1
            v13 = p3 - p1
            l12 = np.linalg.norm(v12)
            l13 = np.linalg.norm(v13)
            gamma = np.arccos(np.dot(v12, v13)/(l12 * l13))
            height =  l13 * np.sin(gamma)
            area = 0.5 * l12 * height

            # perform integral and test it is approximately area * a0
            I = 0.0
            for i,_ in enumerate(verts[:-1]):
                I += mvp.augmented_line_integral(verts[i], verts[i+1])
            I += mvp.augmented_line_integral(verts[-1], verts[0])

            self.assertAlmostEqual(I, a0 * area)

    def test_convex_hull_integral_1(self):
        '''MVP2D.convex_hull_integral: integral over unit cube with shuffled vertices'''

        points = [(0,0), (1,0), (1,1), (0,1)]
        for trial in range(100):

            random.shuffle(points)

            # generate random heigh of uniform curce
            a0 = np.random.uniform(-10, 10)

            # generate random length coefficient mvp
            coef_len = random.choice([1,3,6,10,15,21])
            coefs = np.concatenate(([a0], np.zeros(coef_len-1)))
            mvp = tl.MVP2D(coefs)

            I = mvp.convex_hull_integral(points)

            self.assertAlmostEqual(I, a0)

    def test_convex_hull_integral_2(self):
        '''MVP2D.convex_hull_integral: integral over parabolic dome with random internal points'''

        coefs = np.array([1.0, 0.0, 0.0, -1.0, 0.0, -1.0])
        mvp = tl.MVP2D(coefs)
        b = np.sqrt(2.0)/2.0
        ext_verts = np.asarray([(-b,-b), (b,-b), (b,b), (-b,b)])
        expected_integral = 4.0*b**2 - 8.0/3.0 * b**4

        for trial in range(100):

            points = np.concatenate((ext_verts, np.random.rand(30,2)*2*b-b))
            np.random.shuffle(points)
            I = mvp.convex_hull_integral(points)
            self.assertAlmostEqual(I, expected_integral)

    def test_convex_hull_integral_3(self):
        '''MVP2D.convex_hull_integral: integral over degenerate hull with random mvp'''

        for trial in range(100):

            coefs = np.random.rand(6)*10.0 - 5.0
            mvp = tl.MVP2D(coefs)
            rand_vert_0 = tuple(np.random.rand(2)*20.0 - 10.0)
            rand_vert_1 = tuple(np.random.rand(2)*20.0 - 10.0)
            collinear_vert = tuple(np.random.rand(1)*10.0*np.array([rand_vert_1[0]-rand_vert_0[0], rand_vert_1[1]-rand_vert_0[1]]) + rand_vert_0)
            points = np.array([rand_vert_0, rand_vert_1, collinear_vert])
            np.random.shuffle(points)
            I = mvp.convex_hull_integral(points)
            self.assertAlmostEqual(I, 0.0)

class TestRewardFunctions(unittest.TestCase):
    ''' test RewardFunction class and subclasses
    '''

    def test_poly_reward_func_1(self):
        ''' 2D reward func with random coefs, bounds, and test points
        '''
        c =[7.09386953,
            9.8767273,
            6.89551372,
            -1.50711581,
            5.34473526,
            7.26896898]
        b ={'xmin':-0.39066548, 'xmax':3.65150448, 'ymin':-4.81268648, 'ymax':1.98431945}
        rew_fn = tl.PolynomialRewardFunction2D(coefs=c, bounds=b)
        self.assertAlmostEqual(rew_fn.get_value(0.0, 0.0), 7.09386953)
        self.assertAlmostEqual(rew_fn.get_value(-.39067, 0.0), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(3.6516, 0.0), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(0.0, -4.8127), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(0.0, 1.985), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(3.94626299, -3.76575395), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(1.11495115, 1.94315497), 68.65750655353055)

    def test_radial_polynomial_reward_func_1(self):
        r = 0.79767303
        pv = 1.59169271
        rew_fn = tl.RadialPolynomialRewardFunction2D(radius=r, peak_value=pv)
        self.assertAlmostEqual(rew_fn.get_value(r,0), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(0,r), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(-r,0), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(0,-r), 0.0)
        self.assertAlmostEqual(rew_fn.get_value(0,0), pv)

        x, y = -0.00107664, -0.5515651
        self.assertAlmostEqual(rew_fn.get_value(x,y), 0.8306577242581099)

    def test_radial_polynomial_reward_func_non_negative(self):
        ''' check radial polynomial reward never negative'''
        r = 1.485668841954495
        pv = 0.013853031949074834
        rew_fn = tl.RadialPolynomialRewardFunction2D(radius=r, peak_value=pv)
        for i in range(10000):
            x = np.random.rand()*10 - 5
            y = np.random.rand()*10 - 5
            self.assertTrue(rew_fn.get_value(x,y) >= 0.0)

    def test_extended_radial_polynomial_reward_func_integral_1(self):
        '''ExtendRadialPolynomialRewardFunction2D.get_radial_integral: peak_value=radius=1'''
        for trial in range(100):
            r = np.random.uniform(0, 10)
            rew_fn = tl.ExtendedRadialPolynomialRewardFunction2D(radius=1, peak_value=1)
            positive_region_integral = rew_fn.get_radial_integral(1.0)
            integral = rew_fn.get_radial_integral(r)
            self.assertAlmostEqual(positive_region_integral, 0.5*np.pi)
            self.assertAlmostEqual(integral , np.pi*r**2 * (1.0 - 0.5*r**2))
            self.assertLessEqual(integral, positive_region_integral)

    def test_extended_radial_polynomial_reward_func_integral_2(self):
        '''ExtendRadialPolynomialRewardFunction2D.get_radial_integral: peak_value=1, radius=(random)'''
        for trial in range(100):
            R = np.random.uniform(0, 10)
            r = np.random.uniform(0, 10)
            rew_fn = tl.ExtendedRadialPolynomialRewardFunction2D(radius=R, peak_value=1)
            positive_region_integral = rew_fn.get_radial_integral(R)
            integral = rew_fn.get_radial_integral(r)
            self.assertAlmostEqual(positive_region_integral, 0.5*np.pi*R**2)
            self.assertAlmostEqual(integral , np.pi*r**2/R**2*(R**2 - 0.5*r**2))
            self.assertLessEqual(integral, positive_region_integral)

    def test_extended_radial_polynomial_reward_func_integral_3(self):
        '''ExtendRadialPolynomialRewardFunction2D.get_radial_integral: peak_value=radius=(random)'''
        for trial in range(100):
            R = np.random.uniform(0, 10)
            r = np.random.uniform(0, 10)
            rew_fn = tl.ExtendedRadialPolynomialRewardFunction2D(radius=R, peak_value=R)
            positive_region_integral = rew_fn.get_radial_integral(R)
            integral = rew_fn.get_radial_integral(r)
            self.assertAlmostEqual(positive_region_integral, 0.5*np.pi*R**3)
            self.assertAlmostEqual(integral , np.pi*r**2/R*(R**2 - 0.5*r**2))
            self.assertLessEqual(integral, positive_region_integral)

    # def test_extended_radial_polynomial_reward_func_integral_2(self):
    #     '''ExtendRadialPolynomialRewardFunction2D.get_radial_integral: check integration into negative region decreases value'''
    #     for trial in range(100):
    #         r = np.random.uniform(0, 10)
    #         rew_fn = tl.ExtendedRadialPolynomialRewardFunction2D(radius=r, peak_value=r)
    #         self.assertAlmostEqual(rew_fn.get_radial_integral(r), 0.5*np.pi*r**3)

class TestRiskFunctions(unittest.TestCase):
    ''' test RiskFunction class and subclasses
    '''

    def test_bernoulli_risk_func_1(self):
        ''' 2D risk func with random coefs, bounds, and test points
        '''
        ntrials = 10000
        c =[-0.13688395,
            0.97018536,
            -0.69093378]
        b ={'xmin':-0.50017702, 'xmax':0.494126, 'ymin':-0.68135622, 'ymax':0.46557374}
        risk_fn = tl.BernoulliRiskFunction2D(coefs=c, bounds=b)
        self.assertFalse(risk_fn.sample_failure(-0.51, 0.0))
        self.assertFalse(risk_fn.sample_failure(0.495, 0.0))
        self.assertFalse(risk_fn.sample_failure(0.0,-0.682))
        self.assertFalse(risk_fn.sample_failure(0.0, 0.466))
        # x1, y1 = -0.32011397, -0.18155431
        # count = 0
        # for i in range(ntrials):
        #     count += risk_fn.sample_failure(x1, y1)==True
        # self.assertAlmostEqual(risk_fn.get_failure_probability(x1,y1), 0.0, places=2)

        x2, y2 = 0.16187877, -0.6337289
        count = 0
        for i in range(ntrials):
            count += risk_fn.sample_failure(x2, y2)==True
        self.assertAlmostEqual(risk_fn.get_failure_probability(x2,y2), count/ntrials, places=1)

    def test_radial_bernoulli_risk_func_1(self):
        r = 3.29737487
        risk_fn = tl.RadialBernoulliRiskFunction2D(radius=r)
        self.assertAlmostEqual(risk_fn.get_failure_probability(r,0), 0.0)
        self.assertAlmostEqual(risk_fn.get_failure_probability(0,r), 0.0)
        self.assertAlmostEqual(risk_fn.get_failure_probability(-r,0), 0.0)
        self.assertAlmostEqual(risk_fn.get_failure_probability(0,-r), 0.0)
        self.assertAlmostEqual(risk_fn.get_failure_probability(r,r), 0.0)
        self.assertAlmostEqual(risk_fn.get_failure_probability(-r,r), 0.0)
        self.assertAlmostEqual(risk_fn.get_failure_probability(0,0), 1.0)

        r1 = 0.0273631
        self.assertAlmostEqual(risk_fn.get_failure_probability(r1,0), 0.9999311357300636)
        self.assertAlmostEqual(risk_fn.get_failure_probability(0,r1), 0.9999311357300636)
        self.assertAlmostEqual(risk_fn.get_failure_probability(-r1,0), 0.9999311357300636)
        self.assertAlmostEqual(risk_fn.get_failure_probability(0,-r1), 0.9999311357300636)
        self.assertAlmostEqual(risk_fn.get_failure_probability(r1,r1), 0.9998622714601272)
        self.assertAlmostEqual(risk_fn.get_failure_probability(r1,-r1), 0.9998622714601272)
        self.assertAlmostEqual(risk_fn.get_failure_probability(-r1,r1), 0.9998622714601272)
        self.assertAlmostEqual(risk_fn.get_failure_probability(-r1,-r1), 0.9998622714601272)

        x, y = 0.97943086, 2.40468947
        self.assertAlmostEqual(risk_fn.get_failure_probability(x,y), 0.37993065040754426)

class TestUndirectedGraphCase1(unittest.TestCase):
    ''' test UndirectedGraph class and subclasses
    '''

    def test_undirected_graph_connectivity_1(self):
        ''' UndirectedGraph: full connectivity in 3-node graph with integer keys'''
        net = tl.UndirectedGraph()
        net.add_edge(0, 1)
        net.add_edge(0, 2)
        net.add_edge(1, 2)
        self.assertTrue(net.breadth_first_connectivity_search(0,1))
        self.assertTrue(net.breadth_first_connectivity_search(1,0))
        self.assertTrue(net.breadth_first_connectivity_search(0,2))
        self.assertTrue(net.breadth_first_connectivity_search(2,0))
        self.assertTrue(net.breadth_first_connectivity_search(1,2))
        self.assertTrue(net.breadth_first_connectivity_search(2,1))
        self.assertTrue(net.breadth_first_connectivity_search(0,0))
        self.assertTrue(net.breadth_first_connectivity_search(1,1))
        self.assertTrue(net.breadth_first_connectivity_search(2,2))

    def test_undirected_graph_connectivity_2(self):
        ''' UndirectedGraph: full connectivity in 3-node graph with string keys'''
        net = tl.UndirectedGraph()
        net.add_edge('a', 'b')
        net.add_edge('a', 'c')
        net.add_edge('b', 'c')
        self.assertTrue(net.breadth_first_connectivity_search('a','b'))
        self.assertTrue(net.breadth_first_connectivity_search('b','a'))
        self.assertTrue(net.breadth_first_connectivity_search('a','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','a'))
        self.assertTrue(net.breadth_first_connectivity_search('b','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','b'))


    def test_undirected_graph_connectivity_3(self):
        ''' UndirectedGraph: no connectivity in 3-node graph with integer keys'''
        net = tl.UndirectedGraph()
        net.add_node(0)
        net.add_node(1)
        net.add_node(2)
        self.assertFalse(net.breadth_first_connectivity_search(0,1))
        self.assertFalse(net.breadth_first_connectivity_search(1,0))
        self.assertFalse(net.breadth_first_connectivity_search(0,2))
        self.assertFalse(net.breadth_first_connectivity_search(2,0))
        self.assertFalse(net.breadth_first_connectivity_search(1,2))
        self.assertFalse(net.breadth_first_connectivity_search(2,1))
        self.assertTrue(net.breadth_first_connectivity_search(0,0))
        self.assertTrue(net.breadth_first_connectivity_search(1,1))
        self.assertTrue(net.breadth_first_connectivity_search(2,2))

    def test_undirected_graph_connectivity_4(self):
        ''' UndirectedGraph: partial connectivity in 3-node graph with string keys'''
        net = tl.UndirectedGraph()
        net.add_edge('a', 'b')
        net.add_edge('b', 'c')
        self.assertTrue(net.breadth_first_connectivity_search('a','b'))
        self.assertTrue(net.breadth_first_connectivity_search('b','a'))
        self.assertTrue(net.breadth_first_connectivity_search('a','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','a'))
        self.assertTrue(net.breadth_first_connectivity_search('b','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','b'))

    def test_undirected_graph_connectivity_5(self):
        ''' UndirectedGraph: partially connected 4-node graph with mixed keys'''
        net = tl.UndirectedGraph()
        net.add_edge(0, 'a')
        net.add_edge('a', 1)
        net.add_edge(1, 'b')
        self.assertTrue(net.breadth_first_connectivity_search('a','b'))
        self.assertTrue(net.breadth_first_connectivity_search('b','a'))
        self.assertTrue(net.breadth_first_connectivity_search(0,1))
        self.assertTrue(net.breadth_first_connectivity_search(1,0))
        self.assertTrue(net.breadth_first_connectivity_search(0,'a'))
        self.assertTrue(net.breadth_first_connectivity_search('a',0))
        self.assertTrue(net.breadth_first_connectivity_search(0,'b'))
        self.assertTrue(net.breadth_first_connectivity_search('b',0))
        self.assertTrue(net.breadth_first_connectivity_search(1,'a'))
        self.assertTrue(net.breadth_first_connectivity_search('a',1))
        self.assertTrue(net.breadth_first_connectivity_search(1,'b'))
        self.assertTrue(net.breadth_first_connectivity_search('b',1))

    def test_undirected_graph_connectivity_6(self):
        ''' UndirectedGraph: disconnected graphs with mixed keys'''
        net = tl.UndirectedGraph()
        net.add_edge('a', 'b')
        net.add_edge('b', 'c')
        net.add_edge(0, 1)
        net.add_edge(1, 2)
        self.assertTrue(net.breadth_first_connectivity_search('a','b'))
        self.assertTrue(net.breadth_first_connectivity_search('b','a'))
        self.assertTrue(net.breadth_first_connectivity_search('a','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','a'))
        self.assertTrue(net.breadth_first_connectivity_search('b','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','b'))
        self.assertTrue(net.breadth_first_connectivity_search(0,1))
        self.assertTrue(net.breadth_first_connectivity_search(1,0))
        self.assertTrue(net.breadth_first_connectivity_search(0,2))
        self.assertTrue(net.breadth_first_connectivity_search(2,0))
        self.assertTrue(net.breadth_first_connectivity_search(1,2))
        self.assertTrue(net.breadth_first_connectivity_search(2,1))
        self.assertFalse(net.breadth_first_connectivity_search(0,'a'))
        self.assertFalse(net.breadth_first_connectivity_search('a',0))
        self.assertFalse(net.breadth_first_connectivity_search(0,'b'))
        self.assertFalse(net.breadth_first_connectivity_search('b',0))
        self.assertFalse(net.breadth_first_connectivity_search(0,'c'))
        self.assertFalse(net.breadth_first_connectivity_search('c',0))
        self.assertFalse(net.breadth_first_connectivity_search(1,'a'))
        self.assertFalse(net.breadth_first_connectivity_search('a',1))
        self.assertFalse(net.breadth_first_connectivity_search(1,'b'))
        self.assertFalse(net.breadth_first_connectivity_search('b',1))
        self.assertFalse(net.breadth_first_connectivity_search(1,'c'))
        self.assertFalse(net.breadth_first_connectivity_search('c',1))
        self.assertFalse(net.breadth_first_connectivity_search(2,'a'))
        self.assertFalse(net.breadth_first_connectivity_search('a',2))
        self.assertFalse(net.breadth_first_connectivity_search(2,'b'))
        self.assertFalse(net.breadth_first_connectivity_search('b',2))
        self.assertFalse(net.breadth_first_connectivity_search(2,'c'))
        self.assertFalse(net.breadth_first_connectivity_search('c',2))

    def test_undirected_graph_connectivity_7(self):
        ''' UndirectedGraph: raise exception for checking node not in graph'''
        net = tl.UndirectedGraph()
        net.add_edge('a', 'b')
        net.add_edge('a', 'c')
        self.assertTrue(net.breadth_first_connectivity_search('a','b'))
        self.assertTrue(net.breadth_first_connectivity_search('b','a'))
        self.assertTrue(net.breadth_first_connectivity_search('a','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','a'))
        self.assertTrue(net.breadth_first_connectivity_search('b','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','b'))

        for i in range(100):
            with self.assertRaises(tl.NodeExistenceException):
                net.breadth_first_connectivity_search('a',np.random.randint(100))

    def test_undirected_graph_add_node_1(self):
        ''' UndirectedGraph: ensure that adding a node does not erase previous info '''
        net = tl.UndirectedGraph()
        with self.assertRaises(tl.NodeExistenceException):
            net.breadth_first_connectivity_search('a','b')

        net.add_edge('a', 'b')
        net.add_edge('a', 'c')
        self.assertTrue(net.breadth_first_connectivity_search('a','b'))
        self.assertTrue(net.breadth_first_connectivity_search('b','a'))
        self.assertTrue(net.breadth_first_connectivity_search('a','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','a'))
        self.assertTrue(net.breadth_first_connectivity_search('b','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','b'))

        net.add_node('a')
        self.assertTrue(net.breadth_first_connectivity_search('a','b'))
        self.assertTrue(net.breadth_first_connectivity_search('b','a'))
        self.assertTrue(net.breadth_first_connectivity_search('a','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','a'))
        self.assertTrue(net.breadth_first_connectivity_search('b','c'))
        self.assertTrue(net.breadth_first_connectivity_search('c','b'))


class TestSimpleNetworkCase1(unittest.TestCase):
    ''' test UndirectedGraph class and subclasses
    '''


    class Foo(object):
        ''' hashable object'''
        def __init__(self):
            self.bar = np.random.randint(1)

    def test_simple_network_node_hashing_1(self):
        '''SimpleNetwork: check that hashable and unhashable objects are handled'''
        foo1 = self.Foo()
        foo2 = self.Foo()
        vec1 = np.random.randint(0,10,2) # unhashable
        vec2 = np.random.randint(0,10,2) # unhashable
        nodes = [foo1, 'a', 1, vec1, foo2, 'b', 2, vec2]
        net = tl.SimpleNetwork(nodes)

        self.assertTrue(net.nodes[0] == foo1)
        self.assertTrue(net.nodes[1] == 'a')
        self.assertTrue(net.nodes[2] == 1)
        self.assertTrue(all(net.nodes[3] == vec1))
        self.assertTrue(net.nodes[4] == foo2)
        self.assertTrue(net.nodes[5] == 'b')
        self.assertTrue(net.nodes[6] == 2)
        self.assertTrue(all(net.nodes[7] == vec2))

        # check nodes are only reverenced by index, not keys
        net.add_edge(2,6) # refers to keys 1 and 2
        self.assertTrue(net.breadth_first_connectivity_search(2,6))
        with self.assertRaises(tl.NodeIntegerKeyException):
            net.add_edge('a', 'b')
        with self.assertRaises(tl.NodeIntegerKeyException):
            net.add_edge(foo1, foo2)
        with self.assertRaises(tl.NodeIntegerKeyException):
            net.add_edge(vec1, vec2)



class TestResistanceNetwork(unittest.TestCase):
    ''' test ResistanceNetwork class and subclasses
    '''

    def test_resitance_network_1(self):
        ''' Test laplacian matrix and two-point resistance of 4 node network
        '''

        # network topology: square where opposit nodes 2-4 are directly connect but 1-3 aren't
        # See Wu (2004) Section I; Example 1
        r1, r2 = 8.14799202, 7.90884073
        r_arr = np.array([0, r1, 0, np.inf, r1, 0, r1, r2, r1, 0])
        RN = tl.ResistanceNetwork(4, r_arr)

        c1 = 1.0/r1
        c2 = 1.0/r2

        # test laplacian
        L = RN.laplacian_matrix
        self.assertAlmostEqual(L[0][0], 2*c1)
        self.assertAlmostEqual(L[0][1], -c1)
        self.assertAlmostEqual(L[0][2], 0.0)
        self.assertAlmostEqual(L[0][3], -c1)
        self.assertAlmostEqual(L[1][0], -c1)
        self.assertAlmostEqual(L[1][1], 2*c1 + c2)
        self.assertAlmostEqual(L[1][2], -c1)
        self.assertAlmostEqual(L[1][3], -c2)
        self.assertAlmostEqual(L[2][0], 0.0)
        self.assertAlmostEqual(L[2][1], -c1)
        self.assertAlmostEqual(L[2][2], 2*c1)
        self.assertAlmostEqual(L[2][3], -c1)
        self.assertAlmostEqual(L[3][0], -c1)
        self.assertAlmostEqual(L[3][1], -c2)
        self.assertAlmostEqual(L[3][2], -c1)
        self.assertAlmostEqual(L[3][3], 2*c1+c2)

        # test eigen pairs
        expected_eigenpairs = []
        expected_eigenpairs.append([4*c1, [0.5, -0.5, 0.5, -0.5]])
        expected_eigenpairs.append([2*c1, [-1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2), 0.0]])
        expected_eigenpairs.append([2*c1 + 2*c2, [0.0, -1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2)]])

        eigenpairs = RN.get_nonzero_eigenpairs()
        for i, p in enumerate(eigenpairs):
            self.assertTrue(any(np.isclose(p[0], [ep[0] for ep in expected_eigenpairs])))
            # j = np.where(np.isclose(p[0], [ep[0] for ep in expected_eigenpairs]))[0][0]
            # self.assertAlmostEqual(p[1][0], list(reversed(expected_eigenpairs[j][1]))[0])
            # self.assertAlmostEqual(p[1][1], list(reversed(expected_eigenpairs[j][1]))[1])
            # self.assertAlmostEqual(p[1][2], list(reversed(expected_eigenpairs[j][1]))[2])
            # self.assertAlmostEqual(p[1][3], list(reversed(expected_eigenpairs[j][1]))[3])

        # test resistance
        r02 = (1.0/(4.0*c1))*(0.5-0.5)**2 + (1/(2.0*c1))*(-1.0/np.sqrt(2.0) - 1.0/np.sqrt(2))**2 + (1.0/(2.0*c1 + 2.0*c2))*(0.0-0.0)**2
        r01 = (1.0/(4.0*c1))*(0.5- -0.5)**2 + (1/(2.0*c1))*(-1.0/np.sqrt(2.0) - 0.0)**2 + (1.0/(2.0*c1 + 2.0*c2))*(0.0- -1.0/np.sqrt(2.0))**2
        self.assertAlmostEqual(RN.get_two_point_resistance(0,2), r02)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,1), r01)

    def test_resitance_network_2(self):
        ''' Test two-point resistance of series node network
        '''

        # network topology: series of 6 nodes with random resistors between
        r01, r12, r23, r34, r45 = 3.26096629, 0.90862276, 2.73187024, 3.74048988, 2.29722211
        r_arr = np.array([  0.0,
                            r01, 0.0,
                            np.inf, r12, 0.0,
                            np.inf, np.inf, r23, 0.0,
                            np.inf, np.inf, np.inf, r34, 0.0,
                            np.inf, np.inf, np.inf, np.inf, r45, 0.0])
        RN = tl.ResistanceNetwork(6, r_arr)

        # test two point resistance
        er01 = r01
        er02 = r01 + r12
        er03 = er02 + r23
        er04 = er03 + r34
        er05 = er04 + r45
        er12 = r12
        er13 = er12 + r23
        er14 = er13 + r34
        er15 = er14 + r45
        er23 = r23
        er24 = er23 + r34
        er25 = er24 + r45
        er34 = r34
        er35 = er34 + r45
        er45 = r45
        self.assertAlmostEqual(RN.get_two_point_resistance(0,1), er01)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,2), er02)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,3), er03)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,4), er04)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,5), er05)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,2), er12)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,3), er13)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,4), er14)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,5), er15)
        self.assertAlmostEqual(RN.get_two_point_resistance(2,3), er23)
        self.assertAlmostEqual(RN.get_two_point_resistance(2,4), er24)
        self.assertAlmostEqual(RN.get_two_point_resistance(2,5), er25)
        self.assertAlmostEqual(RN.get_two_point_resistance(3,4), er34)
        self.assertAlmostEqual(RN.get_two_point_resistance(3,5), er35)
        self.assertAlmostEqual(RN.get_two_point_resistance(4,5), er45)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,0), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,1), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(2,2), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(3,3), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(4,4), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(5,5), 0.0)


    def test_resitance_network_3(self):
        ''' Test unconnected network of two nodes
        '''

        # network topology: two unconnected nodes
        r_arr = np.array([0, np.inf, 0])
        RN = tl.ResistanceNetwork(2, r_arr)

        # test resistance is infinite
        self.assertTrue(np.isinf(RN.get_two_point_resistance(0,1)))

    def test_resitance_network_4(self):
        ''' Test two-point resistance of series node network with unconnected node
        '''

        # network topology: series of 6 nodes with random resistors between
        r01, r12, r23, r34, r45 = 3.26096629, 0.90862276, 2.73187024, 3.74048988, np.inf
        r_arr = np.array([  0.0,
                            r01, 0.0,
                            np.inf, r12, 0.0,
                            np.inf, np.inf, r23, 0.0,
                            np.inf, np.inf, np.inf, r34, 0.0,
                            np.inf, np.inf, np.inf, np.inf, r45, 0.0])
        RN = tl.ResistanceNetwork(6, r_arr)

        # test two point resistance
        er01 = r01
        er02 = r01 + r12
        er03 = er02 + r23
        er04 = er03 + r34
        er05 = er04 + r45
        er12 = r12
        er13 = er12 + r23
        er14 = er13 + r34
        er15 = er14 + r45
        er23 = r23
        er24 = er23 + r34
        er25 = er24 + r45
        er34 = r34
        er35 = er34 + r45
        er45 = r45
        self.assertAlmostEqual(RN.get_two_point_resistance(0,1), er01)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,2), er02)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,3), er03)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,4), er04)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,2), er12)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,3), er13)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,4), er14)
        self.assertAlmostEqual(RN.get_two_point_resistance(2,3), er23)
        self.assertAlmostEqual(RN.get_two_point_resistance(2,4), er24)
        self.assertAlmostEqual(RN.get_two_point_resistance(3,4), er34)

        # singular nodes
        self.assertAlmostEqual(RN.get_two_point_resistance(0,0), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(1,1), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(2,2), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(3,3), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(4,4), 0.0)
        self.assertAlmostEqual(RN.get_two_point_resistance(5,5), 0.0)

        # unconnected nodes
        self.assertTrue(np.isinf(RN.get_two_point_resistance(0,5)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(1,5)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(2,5)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(3,5)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(4,5)))


    def test_resitance_network_5(self):
        ''' Test two-point resistance of 6 node, disconnected network
        '''

        # network topology: square where opposite nodes 2-4 are directly connect but 1-3 aren't
        # See Wu (2004) Section I; Example 1
        # two additional nodes are connected to each other but nothing else
        r1, r2, r3 = 8.14799202, 7.90884073, 6.726105639263754
        r_arr = np.array([
            0,
            r1, 0,
            np.inf, r1, 0,
            r1, r2, r1, 0,
            np.inf, np.inf, np.inf, np.inf, 0,
            np.inf, np.inf, np.inf, np.inf, r3, 0])
        RN = tl.ResistanceNetwork(6, r_arr)

        c1 = 1.0/r1
        c2 = 1.0/r2

        # test resistance
        r02 = (1.0/(4.0*c1))*(0.5-0.5)**2 + (1/(2.0*c1))*(-1.0/np.sqrt(2.0) - 1.0/np.sqrt(2))**2 + (1.0/(2.0*c1 + 2.0*c2))*(0.0-0.0)**2
        r01 = (1.0/(4.0*c1))*(0.5- -0.5)**2 + (1/(2.0*c1))*(-1.0/np.sqrt(2.0) - 0.0)**2 + (1.0/(2.0*c1 + 2.0*c2))*(0.0- -1.0/np.sqrt(2.0))**2
        r45 = r3
        self.assertAlmostEqual(RN.get_two_point_resistance(0,2), r02)
        self.assertAlmostEqual(RN.get_two_point_resistance(0,1), r01)
        self.assertAlmostEqual(RN.get_two_point_resistance(4,5), r45)
        self.assertTrue(np.isinf(RN.get_two_point_resistance(0,4)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(1,4)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(2,4)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(3,4)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(0,5)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(1,5)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(2,5)))
        self.assertTrue(np.isinf(RN.get_two_point_resistance(3,5)))


if __name__ == '__main__':
    unittest.main()
