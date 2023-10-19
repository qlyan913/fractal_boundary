SetFactory("OpenCASCADE");

// Define the trapezoid
Point(1) ={0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 2, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Line(1) = {3, 2};
Line(2) = {2, 1};
Line(3) = {1, 4};
Line(4) = {4, 3};

Curve Loop(1) = {1,2,3,4};
Plane Surface(1)={1};

Physical Curve("Right") = {1};
Physical Curve("Bottom") = {2};
Physical Curve("Left") = {3};
Physical Curve("Top") = {4};

Physical Surface("sur")={1};
