//Parameters
plate = 16.2e-3;
h = 2.5e-4; //3e-4;
hx = h/150; //h/100;

//Geometry definition
//Points
Point(1) = {0,0,0,h};
Point(2) = {plate,0,0,h};
Point(3) = {0,plate,0,h};
Point(4) = {plate,plate,0,h};
Point(5) = {R,0,0,hx};
Point(6) = {0,R,0,hx};

//Lines
Circle(7) = {5,1,6};
Line(8) = {6,3};
Line(9) = {3,4};
Line(10) = {4,2};
Line(11) = {2,5};
Line Loop(12) = {7:11};

//Defining surface
Plane Surface(13) = {12};

////Physical elements for boundary conditions
//Physical Line(14) = {7}; //Hole
//Physical Line(15) = {9}; // Top
//Physical Line(16) = {7};
//Physical Line(17) = {7};
//Physical Line(18) = {7};
