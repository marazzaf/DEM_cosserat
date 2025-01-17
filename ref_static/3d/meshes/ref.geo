R = 10.0; // radius
D = 100.0; // cude dimension
refinement = 1.;
f = 5 / refinement; // fine mesh around the inclusion
c = 40 / refinement; // coarse mesh

Point(1) = {0,0,0,f};
Point(2) = {R,0,0,f};
Point(3) = {D,0,0,c};
Point(4) = {D,0,-D,c};
Point(5) = {0,0,-D,c};
Point(6) = {0,0,-R,f};

Point(7) = {0,D,0,c};
Point(8) = {D,D,0,c};
Point(9) = {D,D,-D,c};
Point(10) = {0,D,-D,c};
Point(11) = {0,R,0,f};

Circle(1) = {2,1,6};
Circle(2) = {11,1,2};
Circle(3) = {6,1,11};

Line(4) = {2,3};
Line(5) = {3,4};
Line(6) = {4,5};
Line(7) = {5,6};
Line(8) = {7,8};
Line(9) = {8,9};
Line(10) = {9,10};
Line(11) = {10,7};
Line(12) = {7,11};
Line(13) = {8,3};
Line(14) = {9,4};
Line(15) = {10,5};

Line Loop(100) = {4,5,6,7,-1};
Plane Surface(100) = {100};
Line Loop(101) = {5,-14,-9,13};
Plane Surface(101) = {101};
Line Loop(102) = {14,6,-15,-10};
Plane Surface(102) = {102};
Line Loop(103) = {7,3,-12,-11,15};
Plane Surface(103) = {103};
Line Loop(104) = {8,9,10,11};
Plane Surface(104) = {104};
Line Loop(105) = {4,-13,-8,12,2};
Plane Surface(105) = {105};
Line Loop(106) = {1,3,2};
Plane Surface(106) = {106};

Surface Loop(200) = {106,100,101,102,103,104,105};
Volume(200) = {200};

